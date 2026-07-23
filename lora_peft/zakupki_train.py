from unsloth import FastLanguageModel
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
from transformers import (DataCollatorForSeq2Seq, Trainer,
                          TrainerCallback, TrainingArguments)

from load_dataset import load_train_eval_dataset
from train import TimingCallback, pick_device

p = argparse.ArgumentParser(description='LoRA дообучение модели')
p.add_argument(
    '--model',
    default='Qwen/Qwen3-4B-Instruct-2507',
    help='Путь до папки со скаченными весами модели'
)
args = p.parse_args()

device = pick_device()

# Корень репозитория = родитель каталога lora_peft. Путь к весам не зависит от cwd.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "weights", args.model)

MAX_LEN=768


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=str(MODEL_DIR),
    max_seq_length=MAX_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=False,
    full_finetuning=False
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=42,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

SYSTEM = ("Ты — эксперт по закупочной деятельности. "
          "Тон: деловой, но понятный, без канцеляризмов сверх необходимого. "
          "Не используй фразы вроде «как ИИ, я не могу» — "
          "если ответа нет, просто скажи об этом прямо. "
          "Отвечай на русском.")

DATASET_PATH = os.path.join(ROOT, "data", "zakupki", "dataset.json")
raw = load_train_eval_dataset(DATASET_PATH)

def tokenize(example):
    prompt_msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Вопрос: {example['question']}"}
    ]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": example["answer"]}]

    prompt_text = tokenizer.apply_chat_template(
        prompt_msgs, tokenize=False, add_generation_prompt=True
    )
    full_text = tokenizer.apply_chat_template(
        full_msgs, tokenize=False, add_generation_prompt=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:MAX_LEN]

    n = min(len(prompt_ids), len(full_ids))
    prompt_len = 0
    while prompt_len < n and prompt_ids[prompt_len] == full_ids[prompt_len]:
        prompt_len += 1

    labels = full_ids.copy()
    for i in range(prompt_len):
        labels[i] = -100

    return {"input_ids": full_ids,
            "attention_mask": [1] * len(full_ids),
            "labels": labels}

dataset = raw.map(tokenize, remove_columns=raw["train"].column_names)

collator = DataCollatorForSeq2Seq(
    tokenizer, label_pad_token_id=-100, padding=True
)

training_args = TrainingArguments(
    output_dir="./lora",
    num_train_epochs=2,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    learning_rate=8e-4,
    warmup_ratio=0.03,
    bf16=(device == "cuda"),
    fp16=(device == "mps"),
    group_by_length=True,
    use_liger_kernel=True,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="epoch",
    load_best_model_at_end=False,
    # optim="paged_adamw_8bit" if device == "cuda" else "adamw_torch",
    optim="adamw_8bit",
    report_to="tensorboard",
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

print(f'==Старт обучения модели {model.config._name_or_path}...')
print(model.config)
trainer.train()
print('==Конец обучения, сохранение адаптера...')

model.save_pretrained("./lora-adapter/zakupki")
tokenizer.save_pretrained("./lora-adapter/zakupki")