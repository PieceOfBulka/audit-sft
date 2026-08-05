"""Общие утилиты для обучения и оценки LoRA-адаптеров.

Используется finetune.py, evaluate.py и llm_as_judge.py, чтобы не дублировать
логику между разными датасетами/доменами (audit, zakupki, ...).
"""
import inspect
import json
import os
import time

import torch
from transformers import TrainerCallback, TrainingArguments
from pydantic import BaseModel

# Единый MLflow-эксперимент для всех LoRA-запусков — так все run'ы видны
# рядом в одном дашборде вне зависимости от домена/модели.
MLFLOW_EXPERIMENT = "lora-finetuning"

# Имя файла с метаданными run'а, который сохраняется рядом с адаптером,
# чтобы bertscore.py/llm_as_judge.py могли переоткрыть тот же MLflow run
# (mlflow.start_run(run_id=...)), не зная и не пересобирая гиперпараметры заново.
RUN_META_FILENAME = "mlflow_run.json"


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def silence_max_length_warning(model) -> None:
    """Модели вроде Qwen несут в generation_config.json устаревший max_length
    (общий лимит на весь контекст, не на новые токены) вместе с современным
    max_new_tokens. Раз мы всегда передаём max_new_tokens явно в generate(),
    он и так имеет приоритет — но transformers всё равно печатает
    предупреждение на каждый вызов. Обнуляем max_length один раз при
    загрузке модели: семантически ничего не меняется, просто убирает шум в логе."""
    if getattr(model.generation_config, "max_length", None) is not None:
        model.generation_config.max_length = None


def slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in text)


def build_run_name(domain_or_label: str, model_name: str, rank: int, alpha: int, lr: float) -> str:
    return f"{slug(domain_or_label)}_{slug(model_name)}_r{rank}a{alpha}_lr{lr:g}"


def save_run_meta(adapter_dir: str, run_id: str, run_name: str) -> None:
    """Пишет mlflow_run.json рядом с адаптером — чтобы оценочные скрипты
    могли дописывать метрики в тот же MLflow run, который создал finetune.py."""
    os.makedirs(adapter_dir, exist_ok=True)
    with open(os.path.join(adapter_dir, RUN_META_FILENAME), "w", encoding="utf-8") as fh:
        json.dump({"experiment": MLFLOW_EXPERIMENT, "run_id": run_id, "run_name": run_name},
                  fh, ensure_ascii=False)


def load_run_meta(adapter_dir: str) -> dict | None:
    path = os.path.join(adapter_dir, RUN_META_FILENAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def resolve_model_dir(model: str) -> str:
    """--model может быть либо repo-id вида 'Qwen/Qwen3-4B-Instruct-2507'
    (тогда ищем в weights/<repo-id>), либо уже готовым абсолютным/относительным путём."""
    if os.path.isabs(model) or os.path.isdir(model):
        return model
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(_root, "weights", model)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def list_available_models(max_depth: int = 3) -> list[str]:
    """Сканирует weights/ и возвращает repo-id-подобные пути (относительно
    weights/) для каждой найденной папки с config.json — то, что можно
    передать в --model / resolve_model_dir."""
    weights_root = os.path.join(_repo_root(), "weights")
    found = []
    if not os.path.isdir(weights_root):
        return found
    for dirpath, dirnames, filenames in os.walk(weights_root):
        depth = os.path.relpath(dirpath, weights_root).count(os.sep)
        if depth >= max_depth:
            dirnames[:] = []
            continue
        if "config.json" in filenames:
            found.append(os.path.relpath(dirpath, weights_root))
            dirnames[:] = []  # не спускаемся внутрь весов модели
    return sorted(found)


def list_available_adapters() -> list[str]:
    """Сканирует lora-adapter/ (в корне репо) на предмет папок с adapter_config.json."""
    adapter_root = os.path.join(_repo_root(), "lora-adapter")
    found = []
    if not os.path.isdir(adapter_root):
        return found
    for name in sorted(os.listdir(adapter_root)):
        path = os.path.join(adapter_root, name)
        if os.path.isfile(os.path.join(path, "adapter_config.json")):
            found.append(path)
    return found


def build_training_arguments(*, group_by_length: bool, warmup_ratio: float, **kwargs) -> TrainingArguments:
    """TrainingArguments(**kwargs), но group_by_length/warmup_ratio передаются
    под тем именем, которое реально принимает установленная версия
    transformers — она переименовывает эти параметры между релизами
    (group_by_length -> train_sampling_strategy="group_by_length",
    warmup_ratio -> warmup_step с поддержкой float) без единого цикла
    депрекации на все сразу, так что нельзя полагаться на конкретное имя."""
    ta_params = inspect.signature(TrainingArguments.__init__).parameters

    if "warmup_ratio" in ta_params:
        kwargs["warmup_ratio"] = warmup_ratio
    elif "warmup_step" in ta_params:
        kwargs["warmup_step"] = warmup_ratio  # float = доля шагов
    else:
        print("== предупреждение: не нашёл ни warmup_ratio, ни warmup_step в TrainingArguments")

    if group_by_length:
        if "group_by_length" in ta_params:
            kwargs["group_by_length"] = True
        elif "train_sampling_strategy" in ta_params:
            kwargs["train_sampling_strategy"] = "group_by_length"
        else:
            print("== предупреждение: не нашёл ни group_by_length, ни train_sampling_strategy")

    return TrainingArguments(**kwargs)


class TimingCallback(TrainerCallback):
    """Замер времени: общее, на эпоху и на шаг (optimizer step, без eval)."""

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
        if torch.backends.mps.is_available() and torch.cuda.is_available() is False:
            torch.mps.synchronize()
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
        if eff_batch:
            print(f"  ~на пример:              {avg_step / eff_batch:.2f} с")


def build_user_content(example: dict) -> str:
    """Универсальный сборщик текста пользователя: если у примера есть domain_a/b/c
    (audit-формат) — используем структурированный шаблон, иначе просто вопрос."""
    if all(k in example for k in ("domain_a", "domain_b", "domain_c")):
        return ("Область аудита:\n"
                f"- Направление: {example['domain_a']}\n"
                f"- Задача: {example['domain_b']}\n"
                f"- Стадия процесса: {example['domain_c']}\n"
                f"\nВопрос: {example['question']}")
    return f"Вопрос: {example['question']}"


def make_tokenize_fn(tokenizer, system_prompt: str, max_len: int):
    """Возвращает функцию tokenize(example) -> {input_ids, attention_mask, labels}.

    Маскирование промпта — по общему префиксу токенов между отдельно
    токенизированными prompt_text и full_text (устойчиво к context-dependent
    BPE-токенизации на стыке промпт/ответ, см. историю фиксов в train.py).
    """
    def tokenize(example: dict) -> dict:
        prompt_msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_user_content(example)},
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": example["answer"]}]

        prompt_text = tokenizer.apply_chat_template(
            prompt_msgs, tokenize=False, add_generation_prompt=True
        )
        full_text = tokenizer.apply_chat_template(
            full_msgs, tokenize=False, add_generation_prompt=False
        )
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_len]

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

    return tokenize


DOMAIN_SYSTEM_PROMPTS = {
    "audit": ("Ты — эксперт по внутреннему аудиту и управлению рисками. "
              "Отвечай в профессиональном регистре, используя корректную "
              "терминологию, структурируя рассуждение в логике системы "
              "управления рисками. Отвечай на русском."),
    "zakupki": ("Ты — эксперт по закупочной деятельности. "
                "Тон: деловой, но понятный, без канцеляризмов сверх необходимого. "
                "Не используй фразы вроде «как ИИ, я не могу» — "
                "если ответа нет, просто скажи об этом прямо. "
                "Отвечай на русском."),
}
# easydataset / zakupki5500 — альтернативные версии датасета закупок (другой
# источник/размер), домен и системный промпт у них тот же, что у "zakupki".
DOMAIN_SYSTEM_PROMPTS["easydataset"] = DOMAIN_SYSTEM_PROMPTS["zakupki"]
DOMAIN_SYSTEM_PROMPTS["zakupki5500"] = DOMAIN_SYSTEM_PROMPTS["zakupki"]

DOMAIN_DATASETS = {
    "audit": os.path.join("data", "claude_answers.json"),
    "zakupki": os.path.join("data", "zakupki", "dataset.json"),
    'easydataset': os.path.join("data", "zakupki", "easydataset.json"),
    'zakupki5500': os.path.join("data", "zakupki", "dataset5500.json"),
}


#==================================
# Вопрос и эталонный ответ берутся из реального test-сплита датасета (см.
# llm_as_judge.py), а не выдумываются LLM "из головы" — раньше судья оценивал
# faithfulness ответа к документу, которого нигде в пайплайне не было (ни
# генератор вопроса, ни судья не получали текст регламента), из-за чего
# faithfulness была неверифицируемой метрикой. Теперь судье передаётся
# реальный эталонный ответ из датасета, и faithfulness — это сверка с ним.
JUDGE_PROMPT = (
    "Тебе будут предоставлены вопрос, эталонный ответ (из проверенного датасета) "
    "и ответ тестируемого — сравни ответ тестируемого с эталонным.\n\n"
    "КРИТЕРИИ ОЦЕНКИ (по шкале от 1 до 5):\n"
    "1. Обоснованность (Faithfulness): Насколько ответ тестируемого соответствует "
    "эталонному ответу по фактам. Если в ответе есть факты, противоречащие эталону, "
    "выдуманные детали, ложные ссылки на законы или галлюцинации — снижай балл. "
    "Оценка 5 ставится только если ВСЕ факты в ответе подтверждаются эталонным ответом.\n"
    "2. Полнота (Completeness): Насколько исчерпывающе дан ответ на заданный вопрос "
    "по сравнению с эталонным — учтены ли все важные риски, шаги или условия из эталона?\n\n"
    "3. Лаконичность(Consciousness):  Насколько ответ ясен, логичен и структурирован. "
    "ФОРМАТ ВЫВОДА — СТРОГО ОБЯЗАТЕЛЬНО СОБЛЮДАТЬ:\n"
    "- Верни ОДИН валидный JSON-объект и ничего больше.\n"
    "- НЕ используй markdown-разметку (никаких ```json, ``` или других оберток).\n"
    "- НЕ пиши никакого текста до или после JSON — ни вступления, ни пояснений, ни извинений.\n"
    "- Все четыре поля обязательны, значения *_score — целые числа от 1 до 5, "
    "поле reasoning — непустая строка.\n"
    "- Экранируй кавычки и переносы строк внутри значений строк по правилам JSON "
    "(перенос строки -> \\n, кавычка -> \\\").\n"
    "- Если сомневаешься в формате — лучше выведи упрощённый reasoning, но не нарушай "
    "структуру JSON.\n\n"
    "Ответ должен выглядеть РОВНО так (только с другими значениями):\n"
    "{\"faithfulness_score\": 5, \"consciousness_score\": 3, \"completeness_score\": 4, "
    "\"reasoning\": \"Подробный разбор ответа на русском языке. Перечисли плюсы, минусы, "
    "неточности или пропущенные по сравнению с эталоном детали.\"}"
)

DOMAIN_JUDGE = {
    "audit": ("Ты — главный методолог службы внутреннего аудита и комплаенса крупной корпорации. "
              "Твоя задача — беспристрастно и строго оценить ответ тестируемого аудитора.\n\n") 
              + JUDGE_PROMPT,
    "zakupki": ("Ты — специалист закупочной комиссии с большим стажем в крупной корпорации. "
                "Твоя задача — беспристрастно и строго оценить ответ тестируемого специалиста закупок.\n\n")
                + JUDGE_PROMPT
}
DOMAIN_JUDGE["easydataset"] = DOMAIN_JUDGE["zakupki"]
DOMAIN_JUDGE["zakupki5500"] = DOMAIN_JUDGE["zakupki"]


class Judgement(BaseModel):
    faithfulness_score: int
    consciousness_score: int
    completeness_score: int
    reasoning: str