"""End-to-end MLX LoRA-дообучение Qwen3-4B на Apple Silicon:
данные -> (опц.) 4-bit квантизация -> LoRA-обучение -> fuse в самостоятельную модель.

Запуск из корня репозитория:
    python lora_peft/mlx_finetune.py

Параметры — в блоке «настройки» ниже. Требуется mlx-lm (уже установлен).

ВАЖНО про стабильность: дефолтный learning rate mlx-lm = 1e-5. Значение 1e-4
приводит к расхождению (взрыв лосса). Здесь LR=2e-5 + warmup + cosine —
устойчивый режим.
"""
import json
import math
import os
import random
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_JSON = os.path.join(ROOT, "data", "claude_answers.json")
MLX_DATA_DIR = os.path.join(ROOT, "data", "mlx")
BASE_MODEL = os.path.join(ROOT, "weights", "weights_qwen")
QUANT_MODEL_PATH = os.path.join(ROOT, "weights", "qwen3-4b-mlx-4bit")
ADAPTER_PATH = os.path.join(ROOT, "lora_peft", "mlx-adapter")
FUSED_PATH = os.path.join(ROOT, "weights", "qwen3-4b-audit-mlx")
CONFIG_PATH = os.path.join(ROOT, "lora_peft", "mlx_lora_config.yaml")

# ----------------------------- настройки -----------------------------
QUANTIZE = True            # True: 4-bit база (быстрее/легче по памяти, стабильно). False: fp16 прямо из HF-папки.
EPOCHS = 2
BATCH_SIZE = 4
NUM_LAYERS = 16            # сколько верхних слоёв адаптировать
LEARNING_RATE = 2e-5       # НЕ ставить 1e-4 — разойдётся. Безопасный диапазон 1e-5..3e-5.
WARMUP_STEPS = 60          # линейный разогрев LR (защита от раннего взрыва лосса)
MAX_SEQ_LENGTH = 1024      # покрывает самый длинный пример (~681 токен)
LORA_RANK = 16
LORA_SCALE = 10.0          # масштаб LoRA (в mlx-lm дефолт 20 — высоковато)
LORA_DROPOUT = 0.05
VAL_SPLIT = 0.2
SEED = 42

SYSTEM = ("Ты — эксперт по внутреннему аудиту и управлению рисками. "
          "Отвечай в профессиональном регистре, используя корректную "
          "терминологию, структурируя рассуждение в логике системы "
          "управления рисками. Отвечай на русском.")


def sh(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def step_data():
    """claude_answers.json -> data/mlx/{train,valid}.jsonl в chat-формате (messages)."""
    os.makedirs(MLX_DATA_DIR, exist_ok=True)
    rows = json.load(open(DATA_JSON, encoding="utf-8"))
    records = [{"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": r["question"]},
        {"role": "assistant", "content": r["answer"]},
    ]} for r in rows if r.get("question") and r.get("answer")]
    random.Random(SEED).shuffle(records)
    n_val = int(len(records) * VAL_SPLIT)
    valid, train = records[:n_val], records[n_val:]
    for name, part in (("train", train), ("valid", valid)):
        with open(os.path.join(MLX_DATA_DIR, f"{name}.jsonl"), "w", encoding="utf-8") as f:
            for rec in part:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"данные: train={len(train)}, valid={len(valid)} -> {MLX_DATA_DIR}")
    return len(train)


def step_model():
    """Путь к модели: квантизованной (4-bit) или исходной HF-папке."""
    if not QUANTIZE:
        return BASE_MODEL
    if os.path.exists(QUANT_MODEL_PATH):
        print(f"квантизованная модель уже есть: {QUANT_MODEL_PATH}")
    else:
        sh([sys.executable, "-m", "mlx_lm", "convert", "--hf-path", BASE_MODEL,
            "-q", "--q-bits", "4", "--mlx-path", QUANT_MODEL_PATH])
    return QUANT_MODEL_PATH


def write_config(model_path, iters):
    """YAML-конфиг для `mlx_lm lora --config` со стабильным режимом (warmup + cosine)."""
    cfg = f"""# Авто-сгенерировано mlx_finetune.py
model: "{model_path}"
train: true
data: "{MLX_DATA_DIR}"
adapter_path: "{ADAPTER_PATH}"
fine_tune_type: lora
mask_prompt: true            # лосс только по ответу ассистента
num_layers: {NUM_LAYERS}
batch_size: {BATCH_SIZE}
iters: {iters}
max_seq_length: {MAX_SEQ_LENGTH}
grad_checkpoint: true        # экономия памяти
seed: {SEED}

# валидация — редко и по нескольким батчам, чтобы не тормозить обучение
steps_per_report: 10
steps_per_eval: 200
val_batches: 20
save_every: 200

optimizer: adamw
optimizer_config:
  adamw:
    weight_decay: 0.01

learning_rate: {LEARNING_RATE}
lr_schedule:
  name: cosine_decay
  warmup: {WARMUP_STEPS}
  warmup_init: 1.0e-7
  arguments: [{LEARNING_RATE}, {iters}]

lora_parameters:
  rank: {LORA_RANK}
  scale: {LORA_SCALE}
  dropout: {LORA_DROPOUT}
"""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(cfg)
    print(f"конфиг записан: {CONFIG_PATH}")


def step_train(model_path, n_train):
    iters = math.ceil(n_train / BATCH_SIZE) * EPOCHS
    write_config(model_path, iters)
    sh([sys.executable, "-m", "mlx_lm", "lora", "--config", CONFIG_PATH])
    return iters


def step_fuse(model_path):
    sh([sys.executable, "-m", "mlx_lm", "fuse",
        "--model", model_path, "--adapter-path", ADAPTER_PATH, "--save-path", FUSED_PATH])


if __name__ == "__main__":
    print("== 1/4 подготовка данных")
    n_train = step_data()
    print("== 2/4 модель (квантизация при QUANTIZE=True)")
    model_path = step_model()
    print("== 3/4 LoRA-обучение")
    step_train(model_path, n_train)
    print("== 4/4 fuse адаптера в самостоятельную модель")
    step_fuse(model_path)
    print(f"\nГотово.\n  адаптер:       {ADAPTER_PATH}\n  слитая модель: {FUSED_PATH}")
    print("Проверить генерацию:")
    print(f"  python -m mlx_lm generate --model {FUSED_PATH} "
          f"--system-prompt \"{SYSTEM}\" --prompt \"Что такое риск-аппетит?\"")
