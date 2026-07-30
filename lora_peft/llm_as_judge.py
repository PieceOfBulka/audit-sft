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
from lora_peft.common import DOMAIN_JUDGE, DOMAIN_QUESTIONS, DOMAIN_SYSTEM_PROMPTS, load_run_meta

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_TOKEN'))

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER = os.path.join(_ROOT, "lora-adapter")

# Защитный потолок на суммарную длину ответа оцениваемой модели (догенерация до EOS).
MAX_TOTAL_NEW_TOKENS = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='LLM as a judge оценка')
    p.add_argument("--domain", choices=sorted(DOMAIN_QUESTIONS), default="audit",
                   help="Определяет системный промпт оцениваемой модели и промпты "
                        "для генерации вопроса/судейства (common.py)")
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


def generate_question(question_prompt: str):
    question_generation = client.chat.completions.create(
        model=os.getenv('QUESTION_MODEL_NAME', 'gpt-5-nano'),
        messages=[
            {
                'role': 'system',
                'content': question_prompt
            }
        ]
    )

    return question_generation.choices[0].message.content

def generate_judgement(question: str, reply: str, judge_prompt: str):
    judgement_generation = client.chat.completions.create(
        model=os.getenv('JUDGE_MODEL_NAME', 'gpt-5-nano'),
        messages=[
            {
                'role': 'user',
                'content': question
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
        response_format={ "type": "json_object" }
    )

    return judgement_generation.choices[0].message.content



def main():
    args = parse_args()
    device = pick_device()
    print(f"Устройство: {device}")

    system_prompt = args.system or DOMAIN_SYSTEM_PROMPTS[args.domain]
    question_prompt = DOMAIN_QUESTIONS[args.domain]
    judge_prompt = DOMAIN_JUDGE[args.domain]

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

    print('===Начало цикла llm-as-judge\n')
    iteration = 0
    try:
        while True:
            iteration += 1
            question = generate_question(question_prompt)
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
            judge = generate_judgement(question, reply, judge_prompt)
            judge_time = time.perf_counter() - judge_start
            judge_dict = json.loads(judge)
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

            result_dir = f"judgements/judge_{os.getenv('QUESTION_MODEL_NAME', 'gpt-5-nano')}"
            os.makedirs(result_dir, exist_ok=True)
            result_file = os.path.join(result_dir, args.model.rstrip('/').split('/')[-1])
            if args.lora:
                result_file += f"&&{args.adapter.rstrip('/').split('/')[-1]}"
            result_file += '.txt'
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write('\n==============')
                f.write(f'\n\n==Question\n{question}')
                f.write(f'\n\n==Reply ({reply_time:.1f} с)\n{reply}')
                f.write(f'\n\n==Judgement ({judge_time:.1f} с)\n')
                f.write(f'- faithfulness = {faithfulness}\n')
                f.write(f'- completeness = {completeness}\n')
                f.write(f'- consciousness = {consciousness}\n')
                f.write(f'- Reasoning\n{reasoning}\n')

            is_continue = input('\n====Продолжаем? (y/n): ')
            if is_continue != 'y':
                break
    finally:
        if run_meta:
            trackio.finish()

if __name__=='__main__':
    main()
