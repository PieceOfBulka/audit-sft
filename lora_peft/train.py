import os
import sys

from unsloth import FastLanguageModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time

import torch
from transformers import (AutoTokenizer, DataCollatorForSeq2Seq, Trainer,
                          TrainerCallback, TrainingArguments)

from load_dataset import load_train_eval_dataset
from sft_lora_peft import MODEL_DIR, get_model_with_lora, pick_device

device = pick_device()

MAX_LEN = 768

class TimingCallback(TrainerCallback):
    """Замер времени: общее, на эпоху и на шаг (optimizer step)."""

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start = time.perf_counter()
        self.epoch_start = None
        self.step_start = None
        self.epoch_times = []
        self.step_times = []

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start = time.perf_counter()

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if device == "mps":
            torch.mps.synchronize()  # MPS асинхронен — синхронизируем для точного замера
        if self.step_start is not None:
            self.step_times.append(time.perf_counter() - self.step_start)

    def on_epoch_end(self, args, state, control, **kwargs):
        dt = time.perf_counter() - self.epoch_start
        self.epoch_times.append(dt)
        print(f"[timing] эпоха {len(self.epoch_times)}: {dt / 60:.1f} мин ({dt:.0f} с)")

    def on_train_end(self, args, state, control, **kwargs):
        total = time.perf_counter() - self.train_start
        n_steps = len(self.step_times)
        avg_step = sum(self.step_times) / n_steps if n_steps else 0.0
        avg_epoch = (sum(self.epoch_times) / len(self.epoch_times)
                     if self.epoch_times else 0.0)
        eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        print("\n=== ВРЕМЯ ОБУЧЕНИЯ ===")
        print(f"  Всего (train+eval+save): {total / 60:.1f} мин ({total:.0f} с)")
        print(f"  Среднее на эпоху:        {avg_epoch / 60:.1f} мин ({avg_epoch:.0f} с)")
        print(f"  Всего optimizer-шагов:   {n_steps}")
        print(f"  Среднее на шаг:          {avg_step:.2f} с "
              f"(эффективный батч = {eff_batch} примеров)")
        print(f"  ~на микро-батч:          {avg_step / args.gradient_accumulation_steps:.2f} с "
              f"({args.per_device_train_batch_size} примера)")
        print(f"  ~на пример:              {avg_step / eff_batch:.2f} с")

def main():
    # --- данные: data/claude_answers.json -> сплиты train/test ---
    raw = load_train_eval_dataset()

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
        # lora_dropout=0.05, # будет оптимизация только по памяти
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )


    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    SYSTEM = ("Ты — эксперт по внутреннему аудиту и управлению рисками. "
            "Отвечай в профессиональном регистре, используя корректную "
            "терминологию, структурируя рассуждение в логике системы "
            "управления рисками. Отвечай на русском.")


    def tokenize(example):
        prompt_msgs = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "Область аудита:\n"
                                        f"- Направление: {example['domain_a']}\n"
                                        f"- Задача: {example['domain_b']}\n"
                                        f"- Стадия процесса: {example['domain_c']}\n"
                                        f"\nВопрос: {example['question']}"}
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

        # Границу промпта ищем по общему префиксу, а не по len(prompt_ids):
        # BPE токенизирует текст на стыке промпт/ответ по-разному в зависимости от
        # контекста (context-dependent tokenization), поэтому длины могут не совпасть.
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

    # model = get_model_with_lora(
    #     rank=16,
    #     alpha=32,
    #     lora_dropout=0.05,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
    #                     "gate_proj", "up_proj", "down_proj"],
    # )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        processing_class=tokenizer,
        callbacks=[TimingCallback()],
    )

    print('==Старт обучения...')
    trainer.train()
    print('==Конец обучения, сохранение адаптера...')

    model.save_pretrained("./lora-adapter")
    tokenizer.save_pretrained("./lora-adapter")


if __name__=='__main__':
    main()