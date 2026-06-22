#!/usr/bin/env python3
"""Полный пайплайн LoRA/QLoRA на mlx-lm: данные -> jsonl -> обучение -> адаптер.

Скрипт повторяет логику подготовки данных из lora_peft/train.py, но использует
mlx-lm (Apple Silicon) вместо transformers+peft.

Примеры запуска из корня репозитория:

    # Всё сразу: подготовка jsonl, конвертация весов (если нужно), обучение
    python lora_mlx/train_mlx_lora.py

    # Только подготовить train.jsonl / valid.jsonl
    python lora_mlx/train_mlx_lora.py --prepare-only

    # Обучение на уже подготовленных данных и готовой MLX-модели
    python lora_mlx/train_mlx_lora.py --skip-prepare --skip-convert

    # Конвертация с 4-bit квантизацией (QLoRA)
    python lora_mlx/train_mlx_lora.py --quantize --quantize-bits 4

Рекомендуется запускать обучение с caffeinate, чтобы Mac не усыплял процесс:
    caffeinate -i python lora_mlx/train_mlx_lora.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from load_dataset import load_train_eval_dataset

DEFAULT_JSONL_DIR = ROOT / "data" / "mlx_sft"
DEFAULT_HF_MODEL = ROOT / "weights" / "weights_qwen"
DEFAULT_MLX_MODEL = ROOT / "weights" / "mlx_weights_qwen"
DEFAULT_ADAPTER_DIR = ROOT / "lora_mlx" / "adapter"

SYSTEM = (
    "Ты — эксперт по внутреннему аудиту и управлению рисками. "
    "Отвечай в профессиональном регистре, используя корректную "
    "терминологию, структурируя рассуждение в логике системы "
    "управления рисками. Отвечай на русском."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MLX LoRA: подготовка данных и обучение")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Путь к data/claude_answers.json (по умолчанию — из load_dataset.py)",
    )
    p.add_argument(
        "--jsonl-dir",
        type=Path,
        default=DEFAULT_JSONL_DIR,
        help="Куда писать train.jsonl и valid.jsonl для mlx-lm",
    )
    p.add_argument(
        "--hf-model",
        type=Path,
        default=DEFAULT_HF_MODEL,
        help="Локальные HuggingFace-веса для конвертации в MLX",
    )
    p.add_argument(
        "--mlx-model",
        type=Path,
        default=DEFAULT_MLX_MODEL,
        help="Путь к MLX-модели",
    )
    p.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_DIR,
        help="Куда сохранить обученный адаптер",
    )
    p.add_argument("--max-seq-length", type=int, default=768)

    p.add_argument("--prepare-only", action="store_true", help="Только подготовить jsonl")
    p.add_argument("--convert-only", action="store_true", help="Только конвертировать веса")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-convert", action="store_true")
    p.add_argument("--skip-train", action="store_true")

    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accumulation-steps", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument(
        "--num-layers",
        type=int,
        default=-1,
        help="Число последних слоёв с LoRA (-1 = все слои)",
    )
    p.add_argument("--steps-per-report", type=int, default=10)
    p.add_argument("--steps-per-eval", type=int, default=50)
    p.add_argument("--val-batches", type=int, default=-1)
    p.add_argument("--grad-checkpoint", action="store_true", default=True)
    p.add_argument("--no-grad-checkpoint", action="store_false", dest="grad_checkpoint")
    p.add_argument(
        "--quantize",
        action="store_true",
        help="Квантизировать модель при конвертации HF -> MLX (QLoRA)",
    )
    p.add_argument(
        "--quantize-bits",
        type=int,
        default=4,
        help="Биты квантизации (только с --quantize)",
    )
    p.add_argument(
        "--q-group-size",
        type=int,
        default=64,
        help="group_size квантизации (только с --quantize)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_user_content(example: dict) -> str:
    return (
        "Область аудита:\n"
        f"- Направление: {example['domain_a']}\n"
        f"- Задача: {example['domain_b']}\n"
        f"- Стадия процесса: {example['domain_c']}\n"
        f"\nВопрос: {example['question']}"
    )


def to_chat_record(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": build_user_content(example)},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prepare_jsonl(dataset_path: str | None, out_dir: Path) -> tuple[int, int]:
    """Читает claude_answers.json и пишет train/valid jsonl (тот же сплит, что в PEFT)."""
    splits = load_train_eval_dataset(dataset_path) if dataset_path else load_train_eval_dataset()

    train_rows = []
    for example in splits["train"]:
        if not str(example.get("answer", "")).strip():
            continue
        train_rows.append(to_chat_record(example))

    valid_rows = []
    for example in splits["test"]:
        if not str(example.get("answer", "")).strip():
            continue
        valid_rows.append(to_chat_record(example))

    if len(train_rows) < 2:
        raise ValueError(
            f"Слишком мало train-примеров с ответами ({len(train_rows)}). "
            "Проверьте data/claude_answers.json."
        )

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_jsonl(out_dir / "valid.jsonl", valid_rows)

    print(f"Записано {out_dir / 'train.jsonl'}: {len(train_rows)} примеров")
    print(f"Записано {out_dir / 'valid.jsonl'}: {len(valid_rows)} примеров")
    return len(train_rows), len(valid_rows)


def mlx_model_ready(path: Path) -> bool:
    return path.is_dir() and (path / "config.json").exists()


def convert_hf_to_mlx(
    hf_model: Path,
    mlx_model: Path,
    *,
    quantize: bool,
    q_bits: int,
    q_group_size: int,
) -> None:
    if not hf_model.is_dir():
        raise FileNotFoundError(f"HF-модель не найдена: {hf_model}")

    if mlx_model.exists():
        raise FileExistsError(
            f"Каталог {mlx_model} уже существует. Удалите его или укажите другой --mlx-model."
        )

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm",
        "convert",
        "--hf-path",
        str(hf_model),
        "--mlx-path",
        str(mlx_model),
    ]
    if quantize:
        cmd.extend(["-q", "--q-bits", str(q_bits), "--q-group-size", str(q_group_size)])

    mode = f"{q_bits}-bit QLoRA" if quantize else "float (LoRA)"
    print(f"Конвертация HF -> MLX ({mode}):")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def ensure_mlx_lm_installed() -> None:
    try:
        import mlx_lm  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Пакет mlx-lm не установлен. Выполните:\n"
            '  pip install "mlx-lm[train]"'
        ) from exc


def compute_iters(train_size: int, batch_size: int, grad_accum: int, epochs: int) -> int:
    """Число итераций mlx-lm (микро-батчей), эквивалентных PEFT-эпохам."""
    opt_steps_per_epoch = max(
        1, (train_size + batch_size * grad_accum - 1) // (batch_size * grad_accum)
    )
    return opt_steps_per_epoch * grad_accum * epochs


class TimingCallback:
    """Замер времени обучения MLX LoRA (аналог TimingCallback в lora_peft/train.py)."""

    def __init__(self, *, iters: int, epochs: int, batch_size: int, grad_accum: int):
        self.iters = iters
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.steps_per_epoch = max(1, iters // epochs)
        self.eff_batch = batch_size * grad_accum

        self.train_start = time.perf_counter()
        self.last_epoch_boundary = self.train_start
        self.current_epoch = 0
        self.epoch_times: list[float] = []
        self.val_times: list[float] = []
        self.iter_per_sec: list[float] = []
        self.tokens_per_sec: list[float] = []

    def on_train_loss_report(self, train_info: dict) -> None:
        iteration = train_info["iteration"]
        if it_sec := train_info.get("iterations_per_second"):
            self.iter_per_sec.append(it_sec)
        if tps := train_info.get("tokens_per_second"):
            self.tokens_per_sec.append(tps)

        epoch_idx = min(self.epochs, iteration // self.steps_per_epoch)
        while self.current_epoch < epoch_idx:
            self.current_epoch += 1
            now = time.perf_counter()
            dt = now - self.last_epoch_boundary
            self.epoch_times.append(dt)
            self.last_epoch_boundary = now
            print(f"[timing] эпоха {self.current_epoch}: {dt / 60:.1f} мин ({dt:.0f} с)")

    def on_val_loss_report(self, val_info: dict) -> None:
        if val_time := val_info.get("val_time"):
            self.val_times.append(val_time)

    def print_summary(self) -> None:
        total = time.perf_counter() - self.train_start
        if self.current_epoch < self.epochs:
            dt = time.perf_counter() - self.last_epoch_boundary
            self.epoch_times.append(dt)

        opt_steps = max(1, self.iters // self.grad_accum)
        avg_iter_sec = (1.0 / (sum(self.iter_per_sec) / len(self.iter_per_sec))
                        if self.iter_per_sec else 0.0)
        avg_opt_step = avg_iter_sec * self.grad_accum if avg_iter_sec else 0.0
        avg_epoch = (sum(self.epoch_times) / len(self.epoch_times)
                     if self.epoch_times else 0.0)
        avg_tps = (sum(self.tokens_per_sec) / len(self.tokens_per_sec)
                   if self.tokens_per_sec else 0.0)
        val_total = sum(self.val_times)

        print("\n=== ВРЕМЯ ОБУЧЕНИЯ (MLX) ===")
        print(f"  Всего (train+eval+save): {total / 60:.1f} мин ({total:.0f} с)")
        print(f"  Среднее на эпоху:        {avg_epoch / 60:.1f} мин ({avg_epoch:.0f} с)")
        print(f"  Всего optimizer-шагов: {opt_steps}")
        print(f"  Среднее на шаг:          {avg_opt_step:.2f} с "
              f"(эффективный батч = {self.eff_batch} примеров)")
        if avg_iter_sec:
            print(f"  ~на микро-батч:          {avg_iter_sec:.2f} с "
                  f"({self.batch_size} примера)")
            print(f"  ~на пример:              {avg_opt_step / self.eff_batch:.2f} с")
        if avg_tps:
            print(f"  Средняя скорость:        {avg_tps:.0f} токенов/с")
        if self.val_times:
            print(f"  Eval (суммарно):         {val_total / 60:.1f} мин ({val_total:.0f} с)")


def run_training(args: argparse.Namespace, train_size: int) -> None:
    from mlx_lm.lora import CONFIG_DEFAULTS, train_model
    from mlx_lm.tuner.datasets import load_dataset
    from mlx_lm.utils import load

    iters = compute_iters(
        train_size,
        args.batch_size,
        args.grad_accumulation_steps,
        args.epochs,
    )
    save_every = max(1, iters // args.epochs)

    lora_args = types.SimpleNamespace(
        model=str(args.mlx_model),
        train=True,
        data=str(args.jsonl_dir),
        fine_tune_type="lora",
        optimizer="adamw",
        optimizer_config=CONFIG_DEFAULTS["optimizer_config"],
        mask_prompt=True,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        iters=iters,
        val_batches=args.val_batches,
        learning_rate=args.learning_rate,
        steps_per_report=args.steps_per_report,
        steps_per_eval=args.steps_per_eval,
        grad_accumulation_steps=args.grad_accumulation_steps,
        resume_adapter_file=None,
        adapter_path=str(args.adapter_path),
        save_every=save_every,
        test=False,
        test_batches=500,
        max_seq_length=args.max_seq_length,
        config=None,
        grad_checkpoint=args.grad_checkpoint,
        clear_cache_threshold=0,
        lr_schedule=None,
        lora_parameters={
            "rank": args.lora_rank,
            "dropout": args.lora_dropout,
            "scale": args.lora_alpha,
        },
        report_to=None,
        project_name=None,
        trust_remote_code=False,
        seed=args.seed,
    )

    eff_batch = args.batch_size * args.grad_accumulation_steps
    print("\n=== Параметры обучения MLX LoRA ===")
    print(f"  Модель:              {args.mlx_model}")
    print(f"  Данные:              {args.jsonl_dir}")
    print(f"  Адаптер:             {args.adapter_path}")
    print(f"  Train примеров:      {train_size}")
    print(f"  Эпох:                {args.epochs}")
    print(f"  Итераций (iters):    {iters}")
    print(f"  batch_size:          {args.batch_size}")
    print(f"  grad_accum:          {args.grad_accumulation_steps}")
    print(f"  эффективный батч:    {eff_batch}")
    print(f"  learning_rate:       {args.learning_rate}")
    print(f"  LoRA rank/alpha:     {args.lora_rank}/{args.lora_alpha}")
    print(f"  max_seq_length:      {args.max_seq_length}")
    print(f"  mask_prompt:         True")
    print(f"  grad_checkpoint:     {args.grad_checkpoint}")
    print("== Старт обучения...\n")

    timing = TimingCallback(
        iters=iters,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accumulation_steps,
    )

    print("Загрузка модели...")
    model, tokenizer = load(str(args.mlx_model))
    print("Загрузка датасета...")
    train_set, valid_set, _test_set = load_dataset(lora_args, tokenizer)

    try:
        train_model(lora_args, model, train_set, valid_set, timing)
    finally:
        timing.print_summary()

    adapter_file = args.adapter_path / "adapters.safetensors"
    if adapter_file.exists():
        print(f"\n== Обучение завершено. Адаптер: {adapter_file}")
    else:
        print(f"\n== Обучение завершено. Проверьте каталог: {args.adapter_path}")


def main() -> None:
    args = parse_args()

    only_prepare = args.prepare_only
    only_convert = args.convert_only
    do_prepare = not args.skip_prepare and not only_convert
    do_convert = not args.skip_convert and not only_prepare
    do_train = not args.skip_train and not only_prepare and not only_convert

    train_size = 0

    if do_prepare or only_prepare:
        train_size, valid_size = prepare_jsonl(args.dataset, args.jsonl_dir)
        print(f"Сплит: train={train_size}, valid={valid_size}")
        if only_prepare:
            return

    if do_convert or only_convert:
        ensure_mlx_lm_installed()
        if mlx_model_ready(args.mlx_model):
            print(f"MLX-модель уже есть: {args.mlx_model}")
        else:
            convert_hf_to_mlx(
                args.hf_model,
                args.mlx_model,
                quantize=args.quantize,
                q_bits=args.quantize_bits,
                q_group_size=args.q_group_size,
            )
            print(f"MLX-модель сохранена: {args.mlx_model}")
        if only_convert:
            return

    if do_train:
        ensure_mlx_lm_installed()
        if not (args.jsonl_dir / "train.jsonl").exists():
            raise FileNotFoundError(
                f"Нет {args.jsonl_dir / 'train.jsonl'}. Сначала запустите без --skip-prepare."
            )
        if train_size == 0:
            train_size = sum(1 for _ in (args.jsonl_dir / "train.jsonl").open(encoding="utf-8"))
        if not mlx_model_ready(args.mlx_model):
            raise FileNotFoundError(
                f"MLX-модель не найдена: {args.mlx_model}. "
                "Запустите с конвертацией или укажите --mlx-model."
            )

        if args.adapter_path.exists() and any(args.adapter_path.iterdir()):
            print(f"Внимание: {args.adapter_path} не пуст — mlx-lm допишет/перезапишет адаптер.")
        else:
            args.adapter_path.mkdir(parents=True, exist_ok=True)

        run_training(args, train_size)


if __name__ == "__main__":
    main()
