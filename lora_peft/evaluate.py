#!/usr/bin/env python3
"""Универсальная оценка LoRA-адаптера по BERTScore на любом domain-датасете.

Генерирует ответы модели (base или base+LoRA) на отложенной выборке (test) и
считает BERTScore против эталонных ответов. Обобщает eval_bertscore.py:
раньше система-промпт/датасет/модель были зашиты под audit, теперь — параметры.

Примеры запуска (из корня репозитория):

    python lora_peft/evaluate.py --domain zakupki --model Qwen/Qwen3-8B \\
        --adapter ./lora-adapter/zakupki --num 50

    python lora_peft/evaluate.py --domain zakupki --model Qwen/Qwen3-8B --base-only

Результаты по умолчанию сохраняются в bertscores/<модель>_<датасет>.json
(например bertscores/Qwen_Qwen3-8B_zakupki.json); переопределить путь можно
через --out, либо только каталог через --out-dir.
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import trackio

from load_dataset import load_train_eval_dataset
from lora_peft.common import (DOMAIN_DATASETS, DOMAIN_SYSTEM_PROMPTS,
                               build_user_content, load_run_meta, pick_device,
                               resolve_model_dir, slug)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Универсальная BERTScore-оценка LoRA-адаптера")

    p.add_argument("--domain", choices=sorted(DOMAIN_DATASETS), default=None,
                   help="Готовый профиль (система-промпт + путь к датасету) из common.py")
    p.add_argument("--dataset", default=None, help="Путь к JSON-датасету (переопределяет --domain)")
    p.add_argument("--system-prompt", default=None, help="Переопределяет --domain")

    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="repo-id (ищется в weights/<repo-id>) или готовый путь к весам")
    p.add_argument("--adapter", default=None, help="Каталог с LoRA-адаптером")
    p.add_argument("--base-only", action="store_true", help="Оценить базовую модель без адаптера")

    p.add_argument("--num", type=int, default=50, help="Сколько примеров из test оценивать")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Сколько примеров генерировать за один проход GPU (не путать с train batch)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--lang", default="ru", help="Язык для BERTScore (ru -> mBERT по умолчанию)")
    p.add_argument("--bertscore-model", default=None, help="Переопределить модель BERTScore")
    p.add_argument("--out", default=None,
                   help="Путь к файлу результатов. По умолчанию — "
                        "bertscores/<модель>_<датасет>.json")
    p.add_argument("--out-dir", default="./bertscores",
                   help="Каталог для результатов, если --out не задан явно")

    return p.parse_args()


def default_out_path(out_dir: str, model_name: str, dataset_label: str, adapter_path: str) -> str:
    filename = f"{slug(model_name)}_{slug(dataset_label)}"
    filename += f"_{slug(adapter_path)}" if adapter_path else "_base"
    filename += ".json"
    return os.path.join(out_dir, filename)


def load_model(args, model_dir, device):
    """Пытаемся грузить через Unsloth (быстрый inference-путь, только CUDA),
    иначе — обычный transformers + SDPA-attention (работает везде)."""
    if device == "cuda":
        try:
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_dir,
                max_seq_length=4096,  # с запасом под prompt+max_new_tokens при генерации
                dtype=torch.bfloat16,
                load_in_4bit=False,
            )
            if not args.base_only:
                if not args.adapter:
                    raise SystemExit("Нужен --adapter (или --base-only для оценки без LoRA)")
                from peft import PeftModel
                model = PeftModel.from_pretrained(model, args.adapter)
                print(f"== LoRA-адаптер загружен из {args.adapter} (Unsloth backend)")
            else:
                print("== Оценка базовой модели без адаптера (baseline, Unsloth backend)")

            FastLanguageModel.for_inference(model)  # переключает модель в быстрый inference-режим

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            return model, tokenizer
        except ImportError:
            print("== Unsloth недоступен, откатываюсь на обычный transformers+SDPA")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only генерация: паддинг слева

    dtype = torch.bfloat16 if device == "cuda" else (
        torch.float16 if device == "mps" else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
    )

    if not args.base_only:
        if not args.adapter:
            raise SystemExit("Нужен --adapter (или --base-only для оценки без LoRA)")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"== LoRA-адаптер загружен из {args.adapter}")
    else:
        print("== Оценка базовой модели без адаптера (baseline)")

    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_batch(model, tokenizer, system_prompt, examples, device, max_new_tokens):
    """Генерирует ответы для целого батча примеров за один вызов model.generate.

    Паддинг слева (tokenizer.padding_side='left') гарантирует, что все строки
    в батче заканчивают промпт на одной и той же позиции — поэтому обрезка
    prompt-части (inputs['input_ids'].shape[1]) корректна одинаково для всех строк.
    """
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user", "content": build_user_content(ex)}],
            tokenize=False, add_generation_prompt=True,
        )
        for ex in examples
    ]
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                       add_special_tokens=False).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy -> воспроизводимая оценка
        pad_token_id=tokenizer.pad_token_id,
    )
    gen_ids = out[:, inputs["input_ids"].shape[1]:]
    return [tokenizer.decode(row, skip_special_tokens=True).strip() for row in gen_ids]


def main():
    args = parse_args()

    if args.dataset is None and args.domain is None:
        raise SystemExit("Нужен --domain (встроенный профиль) или --dataset + --system-prompt")

    dataset_path = DOMAIN_DATASETS.get(args.dataset) or args.dataset or DOMAIN_DATASETS.get(args.domain)
    system_prompt = args.system_prompt or (
        DOMAIN_SYSTEM_PROMPTS[args.domain] if args.domain else None
    )
    if system_prompt is None:
        raise SystemExit("Без --domain нужно явно передать --system-prompt")

    device = pick_device()
    model_dir = resolve_model_dir(args.model)
    print(f"== device={device} | model={model_dir}")

    dataset_label = args.domain or os.path.splitext(os.path.basename(dataset_path))[0]
    out_path = args.out or default_out_path(args.out_dir, args.model, dataset_label, args.adapter)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    test = load_train_eval_dataset(dataset_path)["test"]
    n = min(args.num, len(test))
    subset = test.select(range(n))
    print(f"== оцениваем {n} примеров из test ({len(test)} всего)")

    model, tokenizer = load_model(args, model_dir, device)

    preds, refs, questions = [], [], []
    examples = list(subset)
    gen_start = time.perf_counter()
    for start in range(0, len(examples), args.batch_size):
        batch = examples[start:start + args.batch_size]
        batch_preds = generate_batch(model, tokenizer, system_prompt, batch, device,
                                     args.max_new_tokens)
        preds.extend(batch_preds)
        refs.extend(ex["answer"] for ex in batch)
        questions.extend(ex["question"] for ex in batch)
        print(f"  [{min(start + args.batch_size, n)}/{n}] сгенерирован батч ({len(batch)} примеров)")
    generation_time = time.perf_counter() - gen_start

    from bert_score import score as bertscore
    print("== считаем BERTScore (первый запуск скачает модель эмбеддингов)...")
    bertscore_start = time.perf_counter()
    bs_kwargs = {"lang": args.lang}
    if args.bertscore_model:
        bs_kwargs = {"model_type": args.bertscore_model}
    P, R, F1 = bertscore(preds, refs, verbose=True, **bs_kwargs)
    bertscore_time = time.perf_counter() - bertscore_start
    total_eval_time = generation_time + bertscore_time

    result = {
        "num_samples": n,
        "model": args.model,
        "adapter": None if args.base_only else args.adapter,
        "bertscore_model": args.bertscore_model or f"default-for-lang={args.lang}",
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
        "generation_time_sec": round(generation_time, 1),
        "bertscore_time_sec": round(bertscore_time, 1),
        "total_eval_time_sec": round(total_eval_time, 1),
        "sec_per_example": round(generation_time / n, 2) if n else 0.0,
    }
    print("\n=== BERTScore ===")
    print(f"  Precision: {result['precision']}")
    print(f"  Recall:    {result['recall']}")
    print(f"  F1:        {result['f1']}")
    print(f"  Время генерации: {generation_time:.1f} с | BERTScore: {bertscore_time:.1f} с "
          f"| итого: {total_eval_time:.1f} с ({result['sec_per_example']:.2f} с/пример)")

    per_example = [
        {"question": q, "f1": round(f.item(), 4), "prediction": p, "reference": r}
        for q, p, r, f in zip(questions, preds, refs, F1)
    ]
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "examples": per_example}, fh, ensure_ascii=False, indent=2)
    print(f"\n== подробные результаты сохранены в {out_path}")

    if not args.base_only:
        run_meta = load_run_meta(args.adapter)
        if run_meta:
            trackio.init(project=run_meta["project"], name=run_meta["run_name"], resume="must")
            trackio.log({
                "bertscore_precision": result["precision"],
                "bertscore_recall": result["recall"],
                "bertscore_f1": result["f1"],
                "bertscore_generation_time_sec": result["generation_time_sec"],
                "bertscore_time_sec": result["bertscore_time_sec"],
                "bertscore_total_eval_time_sec": result["total_eval_time_sec"],
                "bertscore_sec_per_example": result["sec_per_example"],
            })
            trackio.finish()
            print(f"== метрики дописаны в Trackio-run '{run_meta['run_name']}'")
        else:
            print(f"== {os.path.join(args.adapter, 'trackio_run.json')} не найден — "
                  "пропускаю логирование в Trackio (адаптер обучен без finetune.py?)")


if __name__ == "__main__":
    main()
