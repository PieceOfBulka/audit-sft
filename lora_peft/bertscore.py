#!/usr/bin/env python3
"""Универсальная оценка LoRA-адаптера по BERTScore на любом domain-датасете.

Генерирует ответы модели (base или base+LoRA) на отложенной выборке (test) и
считает BERTScore против эталонных ответов. Обобщает eval_bertscore.py:
раньше система-промпт/датасет/модель были зашиты под audit, теперь — параметры.

Примеры запуска (из корня репозитория):

    python lora_peft/bertscore.py --domain zakupki --model Qwen/Qwen3-8B \\
        --adapter ./lora-adapter/zakupki --num 50

    python lora_peft/bertscore.py --domain zakupki --model Qwen/Qwen3-8B --base-only

Результаты по умолчанию сохраняются в bertscores/<модель>_<датасет>.json
(например bertscores/Qwen_Qwen3-8B_zakupki.json); переопределить путь можно
через --out, либо только каталог через --out-dir. Если рядом с адаптером есть
trackio_run.json (создаётся finetune.py) — метрики и время дописываются в тот
же Trackio-run, что и обучение.
"""
import argparse
import json
import os
import sys
import time

# stdout не привязан к терминалу (запуск в фоне/через nohup/из app.py) ->
# Python буферизует print() блоками в несколько КБ вместо построчного вывода,
# и прогресс не видно, пока буфер не заполнится или процесс не завершится —
# долгая генерация выглядит как зависание, хотя реально работает.
sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import trackio

from load_dataset import load_train_eval_dataset
from lora_peft.common import (DOMAIN_DATASETS, DOMAIN_SYSTEM_PROMPTS, FULL_EVAL_THRESHOLD,
                               TRACKIO_PROJECT, base_run_name, build_user_content,
                               detect_finetune_method, load_run_meta, pick_device,
                               resolve_model_dir, should_log_trackio_avg,
                               silence_max_length_warning, slug)

METHOD_LABELS = {"lora": "LoRA", "qlora": "QLoRA", "full_ft": "Full FT"}


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

    p.add_argument("--eval-on", choices=["test", "train"], default="test",
                   help="test (по умолчанию) — реальное качество/обобщение на невиданных данных. "
                        "train — контрольный прогон на обучающих примерах: если train сильно лучше "
                        "test, это сигнал либо переобучения, либо что test содержит вопросы с "
                        "фактами, которых физически нет в обучающих данных")
    p.add_argument("--num", type=int, default=50, help="Сколько примеров из выбранного сплита оценивать")
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


def default_out_path(out_dir: str, model_name: str, dataset_label: str, adapter_path: str,
                     eval_on: str = "test") -> str:
    filename = f"{slug(model_name)}_{slug(dataset_label)}"
    filename += f"_{slug(adapter_path)}" if adapter_path else "_base"
    if eval_on != "test":
        filename += f"_{eval_on}"
    filename += ".json"
    return os.path.join(out_dir, filename)


def load_model(args, model_dir, device):
    """Пытаемся грузить через Unsloth (быстрый inference-путь, только CUDA),
    иначе — обычный transformers + SDPA-attention (работает везде).

    Ранее казавшееся зависание на этой ветке (FastLanguageModel + сырой
    PeftModel.from_pretrained() поверх него, вместо "родного" для Unsloth
    get_peft_model()) на поверку оказалось в основном буферизацией stdout
    (см. sys.stdout.reconfigure(line_buffering=True) в начале файла и фикс
    построчного чтения в app.py::make_proc_runner) — реальный прогресс шёл,
    просто не долетал до консоли/UI. Раз вывод теперь всегда флашится в
    реальном времени, Unsloth-путь вернули: если он всё же где-то реально
    подвиснет — это будет сразу видно, а не выглядеть немой заморозкой.

    method определяется по имени папки адаптера (detect_finetune_method) —
    QLoRA грузит базу в 4bit (та же точность, что видела модель на обучении),
    Full FT — это чекпоинт всей модели, а не LoRA-адаптер, грузится напрямую
    без PeftModel.

    LoRA/QLoRA-адаптер под Unsloth грузится ОДНИМ вызовом
    FastLanguageModel.from_pretrained(model_name=<адаптер>), а не базой +
    отдельным peft.PeftModel.from_pretrained() поверх — на MoE-моделях
    (эксперты — не nn.Linear, а слитый по всем экспертам параметр) второй
    вариант падает без MoE-патчей Unsloth, которые применяются только
    внутри самого from_pretrained."""
    method = detect_finetune_method(args.adapter) if (args.adapter and not args.base_only) else "lora"

    if device == "cuda":
        try:
            from unsloth import FastLanguageModel

            if method == "full_ft":
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=args.adapter,
                    max_seq_length=4096,
                    dtype=torch.bfloat16,
                    load_in_4bit=False,
                )
                print(f"== Full FT чекпоинт загружен напрямую из {args.adapter} (Unsloth backend)")
            else:
                if not args.base_only and not args.adapter:
                    raise SystemExit("Нужен --adapter (или --base-only для оценки без LoRA)")
                # Один вызов с model_name=<адаптер> (а не сначала база, потом
                # ОТДЕЛЬНЫЙ peft.PeftModel.from_pretrained() поверх неё) — на
                # MoE-моделях (эксперты — не nn.Linear, а параметр, слитый по
                # всем экспертам сразу) второй вариант падает: MoE-специфичные
                # патчи Unsloth применяются только внутри самого from_pretrained,
                # а не сохраняются между двумя отдельными вызовами.
                load_target = model_dir if args.base_only else args.adapter
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=load_target,
                    max_seq_length=4096,  # с запасом под prompt+max_new_tokens при генерации
                    dtype=torch.bfloat16,
                    load_in_4bit=(method == "qlora"),
                )
                if not args.base_only:
                    print(f"== {METHOD_LABELS[method]}-адаптер загружен из {args.adapter} (Unsloth backend)")
                else:
                    print("== Оценка базовой модели без адаптера (baseline, Unsloth backend)")

            FastLanguageModel.for_inference(model)  # переключает модель в быстрый inference-режим
            silence_max_length_warning(model)

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            return model, tokenizer
        except ImportError:
            print("== Unsloth недоступен, откатываюсь на обычный transformers+SDPA")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.adapter if method == "full_ft" else model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # decoder-only генерация: паддинг слева

    dtype = torch.bfloat16 if device == "cuda" else (
        torch.float16 if device == "mps" else torch.float32)

    if method == "full_ft":
        model = AutoModelForCausalLM.from_pretrained(
            args.adapter, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa"
        )
        print(f"== Full FT чекпоинт загружен напрямую из {args.adapter}")
    else:
        quant_kwargs = {}
        if method == "qlora":
            if device != "cuda":
                raise SystemExit("QLoRA-адаптер обучен в 4bit — оценка без Unsloth требует CUDA "
                                 "(bitsandbytes не поддерживает 4bit на CPU/MPS)")
            from transformers import BitsAndBytesConfig
            quant_kwargs = {"quantization_config": BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16), "device_map": {"": 0}}
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="sdpa", **quant_kwargs
        )

        if not args.base_only:
            if not args.adapter:
                raise SystemExit("Нужен --adapter (или --base-only для оценки без LoRA)")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, args.adapter)
            print(f"== {METHOD_LABELS[method]}-адаптер загружен из {args.adapter}")
        else:
            print("== Оценка базовой модели без адаптера (baseline)")

    if method != "qlora":  # bitsandbytes сам разместил веса через device_map, .to() тут упадёт
        model.to(device)
    model.eval()
    silence_max_length_warning(model)
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
            # Qwen3 по умолчанию генерирует <think>...</think> перед ответом
            # (enable_thinking=True) — для оценки нужен сам ответ, а не трейс
            # рассуждений, плюс это резко замедляет генерацию. На чат-шаблонах
            # без этого параметра (не-Qwen3) просто игнорируется.
            enable_thinking=False,
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
    out_path = args.out or default_out_path(args.out_dir, args.model, dataset_label, args.adapter,
                                            eval_on=args.eval_on)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    split = load_train_eval_dataset(dataset_path)[args.eval_on]
    n = min(args.num, len(split))
    subset = split.select(range(n))
    print(f"== оцениваем {n} примеров из {args.eval_on} ({len(split)} всего)")

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
        "eval_on": args.eval_on,
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

    if args.base_only:
        # У базовой модели нет адаптера/trackio_run.json — имя run'а строится
        # детерминированно из модели+домена (base_run_name), чтобы результаты
        # базовой модели тоже попадали на дашборд и были сравнимы с LoRA.
        project = TRACKIO_PROJECT
        run_name = base_run_name(args.model, dataset_label)
        resume_mode = "allow"  # первый прогон создаёт run, следующие — резюмируют
    else:
        run_meta = load_run_meta(args.adapter)
        project = run_meta["project"] if run_meta else None
        run_name = run_meta["run_name"] if run_meta else None
        resume_mode = "must"
        if not run_meta:
            print(f"== {os.path.join(args.adapter, 'trackio_run.json')} не найден — "
                  "пропускаю логирование в Trackio (адаптер обучен без finetune.py?)")

    if run_name:
        # Суффикс _train, чтобы контрольный прогон на обучающих данных не
        # перезаписывал/не путался с основной test-метрикой на том же графике —
        # это два разных числа, которые осмысленно сравнивать бок о бок.
        # Дополнительный _full — когда --num >= FULL_EVAL_THRESHOLD (сплиты
        # обычно в разы больше — гонять на них ВСЕХ непрактично), чтобы
        # полновесная проверка была отдельной, честно сравнимой метрикой,
        # а не смешивалась на графике с быстрыми выборочными проверками.
        suffix = "" if args.eval_on == "test" else f"_{args.eval_on}"
        suffix += "_full" if n >= FULL_EVAL_THRESHOLD else ""
        try:
            trackio.init(project=project, name=run_name, resume=resume_mode)
        except ValueError as exc:
            # resume="must" бросает ValueError, если run с таким именем не
            # существует (см. gradio-app/trackio) — например trackio_run.json
            # у адаптера ссылается на run, которого больше нет (БД пересоздана/
            # очищена). Раньше это валило весь скрипт ДО начала оценки —
            # логирование в Trackio не должно быть жёстким условием для самой
            # оценки, поэтому не падаем, а создаём run заново под тем же именем.
            print(f"== resume=must для run '{run_name}' не удался ({exc}) — "
                  "создаю run заново (resume=allow)")
            trackio.init(project=project, name=run_name, resume="allow")

        f1_key = f"bertscore_f1{suffix}"
        n_key = f"bertscore_num_samples{suffix}"
        if should_log_trackio_avg(project, run_name, n, f1_key, n_key):
            # step=0 ЖЁСТКО + should_log_trackio_avg: без этого повторные
            # прогоны (например --eval-on test и --eval-on train, или
            # просто перезапуски) добавляют новую строку метрики на
            # растущий step, и дашборд рисует линию вместо одного
            # столбика — тот же баг, что был у llm_as_judge.py.
            trackio.log({
                f"bertscore_precision{suffix}": result["precision"],
                f"bertscore_recall{suffix}": result["recall"],
                f1_key: result["f1"],
                n_key: n,
                f"bertscore_generation_time_sec{suffix}": result["generation_time_sec"],
                f"bertscore_time_sec{suffix}": result["bertscore_time_sec"],
                f"bertscore_total_eval_time_sec{suffix}": result["total_eval_time_sec"],
                f"bertscore_sec_per_example{suffix}": result["sec_per_example"],
            }, step=0)
            print(f"== метрики дописаны в Trackio-run '{run_name}'")

        trackio.finish()


if __name__ == "__main__":
    main()
