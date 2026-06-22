#!/usr/bin/env python3
"""Интерактивный чат с моделями

Примеры запуска из корня репозитория:
    python chat_qwen.py
    python chat_qwen.py --lora
    python chat_qwen.py --lora --adapter lora_peft/lora-adapter
    python chat_qwen.py --temperature 0.7 --max-new-tokens 512

Команды в чате:
    /exit, /quit  — выход
    /clear        — сброс истории диалога
    /system TEXT  — задать системный промпт
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lora_peft.sft_lora_peft import MODEL_DIR, pick_device, torch_dtype

DEFAULT_SYSTEM = (
    "Ты — эксперт по внутреннему аудиту и управлению рисками. "
    "Отвечай в профессиональном регистре, используя корректную "
    "терминологию, структурируя рассуждение в логике системы "
    "управления рисками. Отвечай на русском."
)

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER = os.path.join(_ROOT, "lora_peft", "lora-adapter")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Консольный чат с Qwen")
    p.add_argument(
        "--lora",
        action="store_true",
        help="Загрузить LoRA-адаптер поверх базовой модели",
    )
    p.add_argument(
        "--adapter",
        default=DEFAULT_ADAPTER,
        help=f"Путь к LoRA-адаптеру (по умолчанию: {DEFAULT_ADAPTER})",
    )
    p.add_argument(
        "--system",
        default=DEFAULT_SYSTEM,
        help="Системный промпт (можно сменить в чате командой /system)",
    )
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Детерминированная генерация (без сэмплинга)",
    )
    return p.parse_args()


def load_model(use_lora: bool, adapter_path: str, device: str):
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print("Загрузка базовой модели (это может занять минуту)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
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
    messages: list[dict[str, str]],
    device: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    greedy: bool,
) -> str:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    gen_kwargs: dict = {
        "max_new_tokens": max_new_tokens,
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

    output_ids = model.generate(**inputs, **gen_kwargs)
    new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def print_help() -> None:
    print(
        "\nКоманды: /exit — выход, /clear — сброс истории, "
        "/system <текст> — сменить системный промпт\n"
    )


def main() -> None:
    args = parse_args()
    device = pick_device()
    print(f"Устройство: {device}")

    model, tokenizer = load_model(args.lora, args.adapter, device)

    system_prompt = args.system
    history: list[dict[str, str]] = []

    mode = "base + LoRA" if args.lora else "base"
    print(f"\n=== Чат с Qwen ({mode}) ===")
    print_help()

    while True:
        try:
            user_input = input("\n\033[34mВы:\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nДо свидания.")
            break

        if not user_input:
            continue

        lowered = user_input.lower()
        if lowered in {"/exit", "/quit", "/q"}:
            print("До свидания.")
            break
        if lowered == "/clear":
            history.clear()
            print("История диалога очищена.")
            continue
        if lowered.startswith("/system "):
            system_prompt = user_input[len("/system ") :].strip()
            history.clear()
            print("Системный промпт обновлён, история сброшена.")
            continue
        if lowered == "/help":
            print_help()
            continue

        messages = [{"role": "system", "content": system_prompt}, *history]
        messages.append({"role": "user", "content": user_input})

        print("\033[34mМодель:\033[0m ", end="", flush=True)
        try:
            reply = generate_reply(
                model,
                tokenizer,
                messages,
                device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                greedy=args.greedy,
            )
        except Exception as exc:
            print(f"\nОшибка генерации: {exc}", file=sys.stderr)
            continue

        print(reply)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
