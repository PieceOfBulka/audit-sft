"""Оценка дообученного LoRA-адаптера по BERTScore.

Генерирует ответы модели (base + LoRA) на отложенной выборке (test) и считает
BERTScore против эталонных ответов из датасета. BERTScore — метрика семантической
близости генерации к эталону (precision / recall / F1 на эмбеддингах).

Запуск из корня репозитория:
    python lora_peft/eval_bertscore.py                       # 50 примеров, адаптер ./lora-adapter
    python lora_peft/eval_bertscore.py --num 100 --adapter ./lora-adapter
    python lora_peft/eval_bertscore.py --base-only           # baseline без LoRA (для сравнения)
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from load_dataset import load_train_eval_dataset
from sft_lora_peft import MODEL_DIR, pick_device, torch_dtype

# Тот же системный промпт, что и при обучении (lora_peft/train.py) — важно для честной оценки.
SYSTEM = ("Ты — эксперт по внутреннему аудиту и управлению рисками. "
          "Отвечай в профессиональном регистре, используя корректную "
          "терминологию, структурируя рассуждение в логике системы "
          "управления рисками. Отвечай на русском.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter", default="./lora-adapter",
                   help="Каталог с LoRA-адаптером (output из train.py)")
    p.add_argument("--base-only", action="store_true",
                   help="Оценить базовую модель без адаптера (baseline)")
    p.add_argument("--num", type=int, default=50,
                   help="Сколько примеров из test-выборки оценивать (генерация на MPS медленная)")
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--lang", default="ru",
                   help="Язык для BERTScore (выбирает модель по умолчанию: ru -> mBERT)")
    p.add_argument("--bertscore-model", default=None,
                   help="Переопределить модель BERTScore, напр. xlm-roberta-large")
    p.add_argument("--out", default="./bertscore_results.json",
                   help="Куда сохранить метрики и примеры")
    return p.parse_args()


def load_model(args, device):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # для генерации с батчем/одиночно у decoder-only моделей паддинг слева
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch_dtype, low_cpu_mem_usage=True
    )

    if not args.base_only:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"== LoRA-адаптер загружен из {args.adapter}")
    else:
        print("== Оценка базовой модели без адаптера (baseline)")

    model.to(device)
    model.eval()
    return model, tokenizer


@torch.no_grad()
def generate_answer(model, tokenizer, question, device, max_new_tokens):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    prompt_text = tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,  # greedy -> воспроизводимая оценка
        pad_token_id=tokenizer.pad_token_id,
    )
    # отрезаем промпт, декодируем только сгенерированную часть
    gen_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    args = parse_args()
    device = pick_device()
    print(f"== device: {device}")

    test = load_train_eval_dataset()["test"]
    n = min(args.num, len(test))
    subset = test.select(range(n))
    print(f"== оцениваем {n} примеров из test ({len(test)} всего)")

    model, tokenizer = load_model(args, device)

    preds, refs, questions = [], [], []
    for i, ex in enumerate(subset):
        pred = generate_answer(model, tokenizer, ex["question"], device, args.max_new_tokens)
        preds.append(pred)
        refs.append(ex["answer"])
        questions.append(ex["question"])
        print(f"  [{i + 1}/{n}] сгенерирован ответ ({len(pred)} символов)")

    # --- BERTScore ---
    from bert_score import score as bertscore
    print("== считаем BERTScore (первый запуск скачает модель эмбеддингов)...")
    bs_kwargs = {"lang": args.lang}
    if args.bertscore_model:
        # при явной модели lang не используется
        bs_kwargs = {"model_type": args.bertscore_model}
    P, R, F1 = bertscore(preds, refs, verbose=True, **bs_kwargs)

    result = {
        "num_samples": n,
        "adapter": None if args.base_only else args.adapter,
        "bertscore_model": args.bertscore_model or f"default-for-lang={args.lang}",
        "precision": round(P.mean().item(), 4),
        "recall": round(R.mean().item(), 4),
        "f1": round(F1.mean().item(), 4),
    }
    print("\n=== BERTScore ===")
    print(f"  Precision: {result['precision']}")
    print(f"  Recall:    {result['recall']}")
    print(f"  F1:        {result['f1']}")

    # сохраняем метрики + по-примерные F1 и сами тексты для разбора
    per_example = [
        {"question": q, "f1": round(f.item(), 4), "prediction": p, "reference": r}
        for q, p, r, f in zip(questions, preds, refs, F1)
    ]
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"summary": result, "examples": per_example}, fh,
                  ensure_ascii=False, indent=2)
    print(f"\n== подробные результаты сохранены в {args.out}")


if __name__ == "__main__":
    main()
