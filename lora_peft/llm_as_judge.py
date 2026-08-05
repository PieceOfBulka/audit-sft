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

import wandb

from lora_peft.sft_lora_peft import pick_device, torch_dtype
from lora_peft.common import (DOMAIN_DATASETS, DOMAIN_JUDGE, DOMAIN_SYSTEM_PROMPTS, Judgement,
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

    # --- какой сплит и сколько примеров оценивать ---
    p.add_argument("--eval-on", choices=["test", "train"], default="test",
                   help="test (по умолчанию) — реальное качество/обобщение. train — контрольный "
                        "прогон на обучающих примерах (сравнить с test, чтобы понять, не в "
                        "переобучении ли дело или test просто требует незнакомых фактов)")
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
    датасета) — без этого faithfulness была бы неверифицируемой (не с чем сверять).

    system-инструкция идёт ПЕРВЫМ сообщением, а весь оцениваемый материал — ОДНИМ
    user-сообщением, без роли assistant. Раньше reply вставлялся отдельным
    assistant-сообщением в конце списка (после system) — на open-weight чат-шаблонах
    (например minimax-m3 через vLLM/SGLang) это заставляло модель ПРОДОЛЖАТЬ чужую
    реплику вместо генерации новой оценки (сообщение с ролью assistant в конце истории
    воспринимается как недописанный ход, а не как данные для анализа), из-за чего судья
    возвращал очередной ответ на исходный вопрос вместо JSON-оценки."""
    judgement_generation = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': judge_prompt
            },
            {
                'role': 'user',
                'content': (
                    f"Вопрос:\n{question}\n\n"
                    f"Эталонный ответ:\n{reference_answer}\n\n"
                    f"Ответ тестируемого:\n{reply}"
                )
            },
        ],
        # extra_body={
        #     "chat_template_kwargs": {
        #         "enable_thinking": False
        #     },
        #     "guided_json": Judgement.model_json_schema()
        # }
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
    split = load_train_eval_dataset(dataset_path)[args.eval_on]
    if args.shuffle:
        split = split.shuffle(seed=args.seed)
    if args.iterations is not None:
        split = split.select(range(min(args.iterations, len(split))))
    print(f"== вопросы и эталонные ответы берутся из {args.eval_on}-сплита {dataset_path} "
          f"({len(split)} примеров)")

    metric_suffix = "" if args.eval_on == "test" else f"_{args.eval_on}"

    reply_model, reply_tokenizer = load_model(
        use_lora=args.lora,
        model_path=args.model,
        adapter_path=args.adapter,
        device=device
        )

    run_meta = load_run_meta(args.adapter) if args.lora else None
    wandb_run = None
    if run_meta:
        wandb_run = wandb.init(project=run_meta["project"], id=run_meta["run_id"], resume="must")
        # Свой x-metric для построчных оценок ниже — если оставить дефолтный
        # общий step-счётчик run'а, он уже стоит там, где его оставил
        # finetune.py (последний optimizer-шаг), и wandb молча дропает/сливает
        # точки со step меньше текущего, так что линия по iteration=1,2,3...
        # просто не появилась бы. define_metric даёт этим трём метрикам
        # собственную ось, независимую от train/global_step.
        wandb.define_metric("judge_iteration")
        for _name in ("judge_faithfulness", "judge_completeness", "judge_consciousness"):
            wandb.define_metric(f"{_name}{metric_suffix}", step_metric="judge_iteration")
        print(f"== метрики judge будут дописаны в W&B run '{run_meta['run_name']}'")
    elif args.lora:
        print(f"== {os.path.join(args.adapter, 'wandb_run.json')} не найден — "
              "пропускаю логирование в W&B (адаптер обучен без finetune.py?)")

    result_dir = f"judgements/judge_{slug(args.judge_model)}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, args.model.rstrip('/').split('/')[-1])
    if args.lora:
        result_file += f"&&{args.adapter.rstrip('/').split('/')[-1]}"
    if args.eval_on != "test":
        result_file += f"_{args.eval_on}"
    result_file += '.txt'

    print('===Начало цикла llm-as-judge\n')
    iteration = 0
    faithfulness_scores: list[float] = []
    completeness_scores: list[float] = []
    consciousness_scores: list[float] = []
    reply_times: list[float] = []
    judge_times: list[float] = []
    entries: list[str] = []  # готовые блоки по каждому примеру -> пишем в файл одним махом
    try:
        for example in split:
            iteration += 1
            question = build_user_content(example)
            reference_answer = example["answer"]
            print(f'\n--- Пример {iteration}/{len(split)} ---')
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

            faithfulness_scores.append(faithfulness)
            completeness_scores.append(completeness)
            consciousness_scores.append(consciousness)
            reply_times.append(reply_time)
            judge_times.append(judge_time)

            if wandb_run:
                # Построчные оценки — как менялась оценка от примера к примеру
                # внутри ЭТОГО прогона (не путать с judge_*_avg ниже — та метрика
                # для сравнения МЕЖДУ моделями/run'ами, эта — для просмотра
                # разброса ВНУТРИ одного прогона). x-ось — judge_iteration
                # (см. define_metric выше), а не wandb'овский общий step.
                wandb.log({
                    f"judge_faithfulness{metric_suffix}": faithfulness,
                    f"judge_completeness{metric_suffix}": completeness,
                    f"judge_consciousness{metric_suffix}": consciousness,
                    "judge_iteration": iteration,
                })

            entries.append(
                '\n==============' +
                f'\n\n==Question\n{question}' +
                f'\n\n==Reference answer\n{reference_answer}' +
                f'\n\n==Reply ({reply_time:.1f} с)\n{reply}' +
                f'\n\n==Judgement ({judge_time:.1f} с)\n' +
                f'- faithfulness = {faithfulness}\n' +
                f'- completeness = {completeness}\n' +
                f'- consciousness = {consciousness}\n' +
                f'- Reasoning\n{reasoning}\n'
            )

            if args.iterations is None:
                is_continue = input('\n====Продолжаем? (y/n): ')
                if is_continue != 'y':
                    break
    finally:
        n = len(faithfulness_scores)
        if n:
            avg_faithfulness = sum(faithfulness_scores) / n
            avg_completeness = sum(completeness_scores) / n
            avg_consciousness = sum(consciousness_scores) / n

            summary = (
                f"=== СРЕДНИЕ ОЦЕНКИ (n={n} примеров) ===\n"
                f"- faithfulness  = {avg_faithfulness:.2f}\n"
                f"- completeness  = {avg_completeness:.2f}\n"
                f"- consciousness = {avg_consciousness:.2f}\n"
            )
            print(f'\n{summary}')

            # Сводка пишется ДО всех отдельных оценок этого прогона.
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write('\n' + '#' * 60 + '\n')
                f.write(summary)
                f.write('#' * 60 + '\n')
                f.write(''.join(entries))

            if wandb_run:
                # run.summary, а не wandb.log(): одно сравнимое число на run
                # (аналог bar-графика), полностью отдельно от per-example
                # метрик выше (свой ключ, не смешивается со step-линией по
                # judge_iteration). Судью на одном адаптере часто гоняют по
                # нескольку раз — присваивание в summary просто перезаписывает
                # предыдущее значение, а не копит строки истории, так что
                # трекио-style ручная дедупликация (сравнение n со старым
                # прогоном, удаление более старых строк) тут не нужна вовсе.
                wandb_run.summary[f"judge_faithfulness_avg{metric_suffix}"] = round(avg_faithfulness, 3)
                wandb_run.summary[f"judge_completeness_avg{metric_suffix}"] = round(avg_completeness, 3)
                wandb_run.summary[f"judge_consciousness_avg{metric_suffix}"] = round(avg_consciousness, 3)
                wandb_run.summary[f"judge_num_samples{metric_suffix}"] = n
                wandb_run.summary[f"judge_reply_time_avg_sec{metric_suffix}"] = round(sum(reply_times) / n, 2)
                wandb_run.summary[f"judge_eval_time_avg_sec{metric_suffix}"] = round(sum(judge_times) / n, 2)

        if wandb_run:
            if os.path.isfile(result_file):
                artifact = wandb.Artifact(name=f"{run_meta['run_name']}-judge-report{metric_suffix}",
                                          type="report")
                artifact.add_file(result_file)
                wandb.log_artifact(artifact)
                print(f"== репорт судьи сохранён как artifact в W&B: {result_file}")
            wandb.finish()

if __name__=='__main__':
    main()
