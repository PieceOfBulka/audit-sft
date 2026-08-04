import argparse
import os
import sys
import time
import torch
import json
from dotenv import load_dotenv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

import trackio

from lora_peft.sft_lora_peft import pick_device, torch_dtype
from lora_peft.common import (DOMAIN_DATASETS, DOMAIN_JUDGE, DOMAIN_SYSTEM_PROMPTS,
                               build_user_content, load_run_meta,
                               silence_max_length_warning, slug)
from load_dataset import load_train_eval_dataset

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER = os.path.join(_ROOT, "lora-adapter")

# Защитный потолок на суммарную длину ответа оцениваемой модели (догенерация до EOS).
MAX_TOTAL_NEW_TOKENS = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='LLM as a judge оценка')
    p.add_argument("--domain", choices=sorted(DOMAIN_DATASETS), default="audit",
                   help="Определяет системный промпт оцениваемой модели, промпт судьи "
                        "и датасет, из которого берутся вопрос+эталонный ответ (common.py)")
    p.add_argument(
        '--model',
        default='',
        help='Путь к модели, которую хотим оценить'
    )
    p.add_argument(
        '--lora',
        action='store_true',
        help='Загрузить LoRA-адаптер поверх базовой модели'
    )
    p.add_argument(
        '--adapter',
        default=DEFAULT_ADAPTER,
        help='Загрузить LoRA-адаптер поверх базовой модели'
    )
    p.add_argument(
        "--system",
        default=None,
        help="Системный промпт оцениваемой модели (по умолчанию — из --domain)",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="Размер чанка генерации; ответ догенерируется до конца (EOS), не обрезаясь")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Детерминированная генерация (без сэмплинга)",
    )

    # --- бэкенд судьи (OpenAI-совместимый API) ---
    p.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"),
                   help="URL OpenAI-совместимого API. Не задан -> официальный OpenAI. "
                        "Например http://10.246.6.82:8080/v1 для внутреннего сервера")
    p.add_argument("--openai-api-key", default=os.getenv("OPENAI_TOKEN", "not-needed"),
                   help="API-ключ. Внутренние серверы обычно его не проверяют")
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME", "gpt-5-nano"),
                   help="Модель-судья, например minimax-m3")

    # --- сколько примеров из test-сплита датасета оценивать ---
    p.add_argument("--iterations", type=int, default=None,
                   help="Оценить ровно N примеров из test-сплита и выйти без вопроса "
                        "'продолжаем?'. Не задано -> интерактивно по всему test-сплиту "
                        "(спрашивает y/n после каждого примера)")
    p.add_argument("--shuffle", action="store_true",
                   help="Перемешать test-сплит перед выбором примеров (иначе — по порядку)")
    p.add_argument("--seed", type=int, default=42, help="Seed для --shuffle")
    return p.parse_args()

def load_model(use_lora: bool, model_path: str, adapter_path: str, device: str):
    if not os.path.isdir(model_path):
            print(f"Ошибка: модель не найдена: {model_path}", file=sys.stderr)
            sys.exit(1)
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print("Загрузка базовой модели (это может занять минуту)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True
    )

    if use_lora:
        from peft import PeftModel

        if not os.path.isdir(adapter_path):
            print(f"Ошибка: адаптер не найден: {adapter_path}", file=sys.stderr)
            sys.exit(1)
        model = PeftModel.from_pretrained(model, adapter_path)
        print(f"LoRA-адаптер загружен: {adapter_path}")
    else:
        print("Режим: базовая модель без LoRA")

    model.to(device)
    model.eval()
    silence_max_length_warning(model)
    return model, tokenizer

@torch.no_grad()
def generate_reply(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    device: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    greedy: bool,
) -> str:
    tokenized_prompt = tokenizer.apply_chat_template(
        [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': question
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        tokenized_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    gen_kwargs: dict = {
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if greedy:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

    # Полный ответ без обрезки: чанки по max_new_tokens до EOS либо до защитного потолка.
    seq = inputs["input_ids"]
    attn = inputs["attention_mask"]
    new_ids: list[int] = []
    while len(new_ids) < MAX_TOTAL_NEW_TOKENS:
        budget = min(max_new_tokens, MAX_TOTAL_NEW_TOKENS - len(new_ids))
        out = model.generate(
            input_ids=seq, attention_mask=attn, max_new_tokens=budget, **gen_kwargs
        )
        gen = out[0][seq.shape[1]:]
        new_ids.extend(gen.tolist())
        if gen.numel() == 0 or gen[-1].item() == tokenizer.eos_token_id or gen.numel() < budget:
            break
        seq = out
        attn = torch.ones_like(seq)
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_judgement(client: OpenAI, model: str, question: str, reference_answer: str,
                       reply: str, judge_prompt: str):
    """Судья сравнивает reply с reference_answer (реальным ответом из test-сплита
    датасета) — без этого faithfulness была бы неверифицируемой (не с чем сверять)."""
    judgement_generation = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': f"Вопрос:\n{question}\n\nЭталонный ответ:\n{reference_answer}"
            },
            {
                'role': 'assistant',
                'content': reply
            },
            {
                'role': 'system',
                'content': judge_prompt
            }
        ],
        extra_body={
            "chat_template_kwargs": {
                "enable_thinking": False
            }
        }
    )

    return judgement_generation.choices[0].message.content


def parse_judge_json(raw: str) -> dict:
    """Некоторые бэкенды (особенно внутренние, без честной поддержки
    response_format=json_object) всё равно оборачивают JSON в ```json...```
    или добавляют текст вокруг. Вырезаем содержимое между первой '{' и
    последней '}' перед парсингом; если и это не JSON — не роняем весь
    прогон, возвращаем нулевые оценки с сырым текстом в reasoning."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"== ПРЕДУПРЕЖДЕНИЕ: судья вернул невалидный JSON ({exc}), ставлю нулевые оценки")
        return {
            "faithfulness_score": 0,
            "completeness_score": 0,
            "consciousness_score": 0,
            "reasoning": f"[невалидный JSON от судьи] {raw[:2000]}",
        }


def main():
    args = parse_args()
    device = pick_device()
    print(f"Устройство: {device}")
    print(f"== judge backend: {args.openai_base_url or 'https://api.openai.com/v1 (по умолчанию)'} "
          f"| judge_model={args.judge_model}")
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url or None)

    system_prompt = args.system or DOMAIN_SYSTEM_PROMPTS[args.domain]
    judge_prompt = DOMAIN_JUDGE[args.domain]

    dataset_path = DOMAIN_DATASETS[args.domain]
    test = load_train_eval_dataset(dataset_path)["test"]
    if args.shuffle:
        test = test.shuffle(seed=args.seed)
    if args.iterations is not None:
        test = test.select(range(min(args.iterations, len(test))))
    print(f"== вопросы и эталонные ответы берутся из test-сплита {dataset_path} "
          f"({len(test)} примеров)")

    reply_model, reply_tokenizer = load_model(
        use_lora=args.lora,
        model_path=args.model,
        adapter_path=args.adapter,
        device=device
        )

    run_meta = load_run_meta(args.adapter) if args.lora else None
    if run_meta:
        trackio.init(project=run_meta["project"], name=run_meta["run_name"], resume="must")
        print(f"== метрики judge будут дописаны в Trackio-run '{run_meta['run_name']}'")
    elif args.lora:
        print(f"== {os.path.join(args.adapter, 'trackio_run.json')} не найден — "
              "пропускаю логирование в Trackio (адаптер обучен без finetune.py?)")

    result_dir = f"judgements/judge_{slug(args.judge_model)}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, args.model.rstrip('/').split('/')[-1])
    if args.lora:
        result_file += f"&&{args.adapter.rstrip('/').split('/')[-1]}"
    result_file += '.txt'

    print('===Начало цикла llm-as-judge\n')
    iteration = 0
    try:
        for example in test:
            iteration += 1
            question = build_user_content(example)
            reference_answer = example["answer"]
            print(f'\n--- Пример {iteration}/{len(test)} ---')
            print(f'=Вопрос:\n{question}')

            reply_start = time.perf_counter()
            reply = generate_reply(
                model=reply_model,
                tokenizer=reply_tokenizer,
                question=question,
                system_prompt=system_prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                greedy=args.greedy)
            reply_time = time.perf_counter() - reply_start
            print(f'=Ответ модели ({reply_time:.1f} с):\n{reply}')

            judge_start = time.perf_counter()
            judge = generate_judgement(client, args.judge_model, question, reference_answer,
                                       reply, judge_prompt)
            judge_time = time.perf_counter() - judge_start
            judge_dict = parse_judge_json(judge)
            reasoning = judge_dict.get("reasoning", "Описание отсутствует")
            faithfulness = judge_dict.get("faithfulness_score", 0)
            completeness = judge_dict.get("completeness_score", 0)
            consciousness = judge_dict.get("consciousness_score", 0)
            print(f'=Оценка судьи ({judge_time:.1f} с):\n{json.dumps(judge_dict,indent=2,ensure_ascii=False)}')

            if run_meta:
                trackio.log({
                    "judge_faithfulness": faithfulness,
                    "judge_completeness": completeness,
                    "judge_consciousness": consciousness,
                    "judge_reply_time_sec": round(reply_time, 1),
                    "judge_eval_time_sec": round(judge_time, 1),
                }, step=iteration)

            with open(result_file, 'a', encoding='utf-8') as f:
                f.write('\n==============')
                f.write(f'\n\n==Question\n{question}')
                f.write(f'\n\n==Reference answer\n{reference_answer}')
                f.write(f'\n\n==Reply ({reply_time:.1f} с)\n{reply}')
                f.write(f'\n\n==Judgement ({judge_time:.1f} с)\n')
                f.write(f'- faithfulness = {faithfulness}\n')
                f.write(f'- completeness = {completeness}\n')
                f.write(f'- consciousness = {consciousness}\n')
                f.write(f'- Reasoning\n{reasoning}\n')

            if args.iterations is None:
                is_continue = input('\n====Продолжаем? (y/n): ')
                if is_continue != 'y':
                    break
    finally:
        if run_meta:
            if os.path.isfile(result_file):
                trackio.log_artifact(result_file, name=f"{run_meta['run_name']}-judge-report",
                                     type="report")
                print(f"== репорт судьи сохранён как artifact в Trackio: {result_file}")
            trackio.finish()

if __name__=='__main__':
    main()
