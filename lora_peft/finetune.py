#!/usr/bin/env python3
"""Универсальный CLI для LoRA-дообучения (Unsloth) на любом domain-датасете.

Заменяет отдельные train.py / zakupki_train.py — вместо копирования файла под
новый датасет просто передаёшь параметры.

Примеры запуска (из корня репозитория):

    # встроенный домен (system prompt + путь к датасету берутся из common.py)
    CUDA_VISIBLE_DEVICES=0 python lora_peft/finetune.py --domain zakupki \\
        --model Qwen/Qwen3-8B --rank 16 --alpha 32 --lr 2e-4 \\
        --batch-size 8 --grad-accum-steps 4

    # свой датасет/промпт, не заведённый в common.py
    python lora_peft/finetune.py \\
        --dataset data/my_domain/dataset.json \\
        --system-prompt "Ты — ассистент по теме X. Отвечай на русском." \\
        --model Qwen/Qwen3-4B-Instruct-2507 \\
        --adapter-name my_domain --rank 8 --alpha 16 --lr 1e-4
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unsloth import FastLanguageModel  # должен импортироваться первым (патчит transformers)

import torch
from transformers import DataCollatorForSeq2Seq, Trainer
from dotenv import load_dotenv

import wandb

from load_dataset import load_train_eval_dataset
from lora_peft.common import (DOMAIN_DATASETS, DOMAIN_SYSTEM_PROMPTS,
                               WANDB_PROJECT, TimingCallback, build_run_name,
                               build_training_arguments, make_tokenize_fn,
                               pick_device, resolve_model_dir,
                               save_run_meta, slug)

# Без этого WANDB_API_KEY/WANDB_ENTITY из .env не попадают в os.environ при
# прямом запуске `python lora_peft/finetune.py` (через app.py сработало бы
# само, т.к. app.py грузит .env и передаёт своё окружение в subprocess).
load_dotenv()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Универсальное LoRA-дообучение (Unsloth)")

    # --- датасет / домен ---
    p.add_argument("--domain", choices=sorted(DOMAIN_DATASETS), default=None,
                   help="Готовый профиль (система-промпт + путь к датасету) из common.py")
    p.add_argument("--dataset", default=None,
                   help="Путь к JSON-датасету (переопределяет --domain, обязателен без --domain)")
    p.add_argument("--system-prompt", default=None,
                   help="Системный промпт (переопределяет --domain, обязателен без --domain)")

    # --- модель ---
    p.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507",
                   help="repo-id (ищется в weights/<repo-id>) или готовый путь к весам")
    p.add_argument("--max-seq-length", type=int, default=768)

    # --- LoRA-гиперпараметры ---
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.0,
                   help="0.0 включает fused-путь Unsloth (быстрее); >0 отключает его, но даёт регуляризацию")
    p.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Через запятую. Например 'q_proj,v_proj' — только attention, безопаснее для узких датасетов")

    # --- обучение ---
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=8,
                   help="per_device_train/eval_batch_size")
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.03)
    p.add_argument("--group-by-length", action="store_true", default=True)
    p.add_argument("--no-group-by-length", action="store_false", dest="group_by_length")
    p.add_argument("--liger", action="store_true", default=True,
                   help="use_liger_kernel в TrainingArguments")
    p.add_argument("--no-liger", action="store_false", dest="liger")
    p.add_argument("--eval-steps", type=int, default=50)
    p.add_argument("--load-best-model-at-end", action="store_true", default=False)

    # --- вывод ---
    p.add_argument("--output-dir", default="./lora")
    p.add_argument("--adapter-name", default=None,
                   help="Подкаталог в ./lora-adapter/<name>. По умолчанию — --domain или 'default'")

    return p.parse_args()

def default_adapter_path(model_name: str, dataset_label: str, adapter_name: str) -> str:
    if adapter_name:
        return os.path.join('./lora-adapter', adapter_name)
    filename = f"{slug(model_name)}_{slug(dataset_label)}_adapter"
    return os.path.join('./lora-adapter', filename)


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

    adapter_dir = default_adapter_path(model_name=args.model, dataset_label=dataset_path, adapter_name=args.adapter_name)

    device = pick_device()
    model_dir = resolve_model_dir(args.model)
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]

    run_label = args.adapter_name or args.domain or os.path.basename(adapter_dir)
    run_name = build_run_name(run_label, args.model, args.rank, args.alpha, args.epochs, args.lr)

    # WandbCallback (report_to="wandb") запускает run сам только если
    # wandb.run is None (см. WandbCallback.setup в transformers/integrations)
    # — раз мы уже открыли run здесь заранее, колбэк просто подключится к
    # нему и НЕ станет закрывать его в on_train_end (в отличие от Trackio,
    # у WandbCallback такого вызова нет вовсе), поэтому держим ссылку до
    # конца main() и закрываем сами.
    run = wandb.init(project=WANDB_PROJECT, name=run_name)

    print(f"== device={device} | model={model_dir} | dataset={dataset_path}")
    print(f"== LoRA: rank={args.rank} alpha={args.alpha} dropout={args.lora_dropout} "
          f"target_modules={target_modules}")
    print(f"== batch={args.batch_size} grad_accum={args.grad_accum_steps} "
          f"(эффективный={args.batch_size * args.grad_accum_steps}) lr={args.lr}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=args.max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        full_finetuning=False,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.rank,
        target_modules=target_modules,
        lora_alpha=args.alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw = load_train_eval_dataset(dataset_path)
    tokenize = make_tokenize_fn(tokenizer, system_prompt, args.max_seq_length)
    dataset = raw.map(tokenize, remove_columns=raw["train"].column_names)

    collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, padding=True)

    training_args = build_training_arguments(
        group_by_length=args.group_by_length,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        bf16=(device == "cuda"),
        fp16=(device == "mps"),
        use_liger_kernel=args.liger,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="epoch",
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model="eval_loss" if args.load_best_model_at_end else None,
        optim="adamw_8bit",
        report_to="wandb",
        run_name=run_name,
        dataloader_pin_memory=(device == "cuda"),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[TimingCallback()],
    )

    print(f"== Старт обучения модели {model.config._name_or_path}...")
    trainer.train()
    print(f"== Обучение завершено, сохраняю адаптер в {adapter_dir}")

    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Метаданные run'а — чтобы evaluate.py/llm_as_judge.py дописывали метрики
    # оценки (bertscore, judge, время) в тот же W&B run, не заводя новый.
    save_run_meta(adapter_dir, run.id, run_name)

    # run не был закрыт WandbCallback (мы его открыли заранее) — прикладываем
    # адаптер как artifact и закрываем run сами.
    artifact = wandb.Artifact(name=f"{run_name}-adapter", type="model")
    artifact.add_dir(adapter_dir)
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    main()
