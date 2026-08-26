import argparse
import os
import sys
import time
import torch
import json
from dotenv import load_dotenv

# stdout не привязан к терминалу (фон/nohup/из app.py) -> Python буферизует
# print() блоками вместо построчного вывода, и прогресс не видно, пока буфер
# не заполнится или процесс не завершится (см. тот же баг у bertscore.py).
sys.stdout.reconfigure(line_buffering=True)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM

import trackio

from lora_peft.sft_lora_peft import pick_device, torch_dtype
from lora_peft.common import (DOMAIN_DATASETS, DOMAIN_JUDGE, DOMAIN_SYSTEM_PROMPTS,
                               FULL_EVAL_THRESHOLD, TRACKIO_PROJECT, Judgement, base_run_name,
                               build_user_content, detect_finetune_method, load_run_meta,
                               should_log_trackio_avg, silence_max_length_warning, slug)
from load_dataset import load_train_eval_dataset

load_dotenv()

_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ADAPTER = os.path.join(_ROOT, "lora-adapter")

# Защитный потолок на суммарную длину ответа оцениваемой модели (догенерация до EOS).
MAX_TOTAL_NEW_TOKENS = 8192


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='LLM as a judge оценка')
    p.add_argument("--domain", choices=sorted(DOMAIN_DATASETS), default="audit",
                   help="Определяет системный промпт оцениваемой модели, промпт судьи "
                        "и датасет, из которого берутся вопрос+эталонный ответ (common.py)")
    p.add_argument(
        '--model',
        default='',
        help='Путь к модели, которую хотим оценить'
    )
    p.add_argument(
        '--lora',
        action='store_true',
        help='Загрузить LoRA-адаптер поверх базовой модели'
    )
    p.add_argument(
        '--adapter',
        default=DEFAULT_ADAPTER,
        help='Загрузить LoRA-адаптер поверх базовой модели'
    )
    p.add_argument(
        "--system",
        default=None,
        help="Системный промпт оцениваемой модели (по умолчанию — из --domain)",
    )
    p.add_argument("--max-new-tokens", type=int, default=1024,
                   help="Размер чанка генерации; ответ догенерируется до конца (EOS), не обрезаясь")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument(
        "--greedy",
        action="store_true",
        help="Детерминированная генерация (без сэмплинга)",
    )

    # --- бэкенд судьи (OpenAI-совместимый API) ---
    p.add_argument("--openai-base-url", default=os.getenv("OPENAI_BASE_URL"),
                   help="URL OpenAI-совместимого API. Не задан -> официальный OpenAI. "
                        "Например http://10.246.6.82:8080/v1 для внутреннего сервера")
    p.add_argument("--openai-api-key", default=os.getenv("OPENAI_TOKEN", "not-needed"),
                   help="API-ключ. Внутренние серверы обычно его не проверяют")
    p.add_argument("--judge-model", default=os.getenv("JUDGE_MODEL_NAME", "gpt-5-nano"),
                   help="Модель-судья, например minimax-m3")

    # --- какой сплит и сколько примеров оценивать ---
    p.add_argument("--eval-on", choices=["test", "train"], default="test",
                   help="test (по умолчанию) — реальное качество/обобщение. train — контрольный "
                        "прогон на обучающих примерах (сравнить с test, чтобы понять, не в "
                        "переобучении ли дело или test просто требует незнакомых фактов)")
    p.add_argument("--iterations", type=int, default=None,
                   help="Оценить ровно N примеров из test-сплита и выйти без вопроса "
                        "'продолжаем?'. Не задано -> интерактивно по всему test-сплиту "
                        "(спрашивает y/n после каждого примера)")
    p.add_argument("--shuffle", action="store_true",
                   help="Перемешать test-сплит перед выбором примеров (иначе — по порядку)")
    p.add_argument("--seed", type=int, default=42, help="Seed для --shuffle")
    return p.parse_args()

METHOD_LABELS = {"lora": "LoRA", "qlora": "QLoRA", "full_ft": "Full FT"}


def load_model(use_lora: bool, model_path: str, adapter_path: str, device: str):
    # method определяется по имени папки адаптера (см. common.detect_finetune_method
    # и finetune.py::default_adapter_path): QLoRA грузит базу в 4bit — та же
    # точность, что видела модель на обучении; Full FT — это чекпоинт всей
    # модели, а не LoRA-адаптер, грузится напрямую без PeftModel.
    method = detect_finetune_method(adapter_path) if use_lora else "lora"

    if method == "full_ft":
        if not os.path.isdir(adapter_path):
            print(f"Ошибка: адаптер не найден: {adapter_path}", file=sys.stderr)
            sys.exit(1)
        print("Загрузка Full FT чекпоинта (это может занять минуту)...")
        tokenizer = AutoTokenizer.from_pretrained(adapter_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(
            adapter_path, dtype=torch_dtype, low_cpu_mem_usage=True
        )
        print(f"Full FT чекпоинт загружен напрямую: {adapter_path}")
        model.to(device)
        model.eval()
        return model, tokenizer

    if not os.path.isdir(model_path):
            print(f"Ошибка: модель не найдена: {model_path}", file=sys.stderr)
            sys.exit(1)
    print("Загрузка токенизатора...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    print("Загрузка базовой модели (это может занять минуту)...")
    quant_kwargs = {}
    if method == "qlora":
        if device != "cuda":
            print("Ошибка: QLoRA-адаптер обучен в 4bit — оценка требует CUDA "
                  "(bitsandbytes не поддерживает 4bit на CPU/MPS)", file=sys.stderr)
            sys.exit(1)
        from transformers import BitsAndBytesConfig
        quant_kwargs = {"quantization_config": BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype), "device_map": {"": 0}}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        **quant_kwargs,
    )

    if use_lora:
        from peft import PeftModel

        if not os.path.isdir(adapter_path):
            print(f"Ошибка: адаптер не найден: {adapter_path}", file=sys.stderr)
            sys.exit(1)
        try:
            model = PeftModel.from_pretrained(model, adapter_path)
            print(f"{METHOD_LABELS[method]}-адаптер загружен: {adapter_path}")
        except (ValueError, RuntimeError) as exc:
            # MoE-модели (например Qwen3-30B-A3B) обучаются через
            # Unsloth.FastLanguageModel.get_peft_model(), который навешивает
            # LoRA на mlp.experts своими патчами поверх peft — голый peft
            # этого не умеет (experts там не nn.Linear, а параметр, слитый
            # по всем экспертам сразу), отсюда и ValueError выше. Один вызов
            # FastLanguageModel.from_pretrained(model_name=<путь к адаптеру>)
            # сам детектит LoRA/QLoRA-адаптер и грузит база+адаптер вместе с
            # теми же патчами (в отличие от связки "модель через Unsloth +
            # отдельный PeftModel.from_pretrained()", которая зависала —
            # см. edf07d3) — поэтому только на CUDA и только как fallback,
            # не трогаем уже рабочий путь для обычных dense-моделей.
            if device != "cuda":
                raise
            print(f"== голый PeftModel.from_pretrained() упал ({exc}), "
                  "пробую через Unsloth (похоже на MoE-архитектуру)")
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            from unsloth import FastLanguageModel

            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=adapter_path,
                max_seq_length=4096,  # с запасом под prompt+max_new_tokens при генерации
                dtype=torch_dtype,
                load_in_4bit=(method == "qlora"),
            )
            FastLanguageModel.for_inference(model)  # быстрый inference-режим
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = 'left'
            print(f"== {METHOD_LABELS[method]}-адаптер загружен через Unsloth: {adapter_path}")
    else:
        print("Режим: базовая модель без LoRA")

    if method != "qlora":  # bitsandbytes сам разместил веса через device_map, .to() тут упадёт
        model.to(device)
    model.eval()
    silence_max_length_warning(model)
    return model, tokenizer

@torch.no_grad()
def generate_reply(
    model,
    tokenizer,
    question: str,
    system_prompt: str,
    device: str,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    greedy: bool,
) -> str:
    tokenized_prompt = tokenizer.apply_chat_template(
        [
            {
                'role': 'system',
                'content': system_prompt
            },
            {
                'role': 'user',
                'content': question
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3 по умолчанию генерирует <think>...</think> перед ответом
        # (enable_thinking=True) — для оценки нужен сам ответ, а не трейс
        # рассуждений, плюс это резко замедляет генерацию. На чат-шаблонах
        # без этого параметра (не-Qwen3) просто игнорируется.
        enable_thinking=False,
    )
    inputs = tokenizer(
        tokenized_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    ).to(device)

    gen_kwargs: dict = {
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

    # Полный ответ без обрезки: чанки по max_new_tokens до EOS либо до защитного потолка.
    seq = inputs["input_ids"]
    attn = inputs["attention_mask"]
    new_ids: list[int] = []
    while len(new_ids) < MAX_TOTAL_NEW_TOKENS:
        budget = min(max_new_tokens, MAX_TOTAL_NEW_TOKENS - len(new_ids))
        out = model.generate(
            input_ids=seq, attention_mask=attn, max_new_tokens=budget, **gen_kwargs
        )
        gen = out[0][seq.shape[1]:]
        new_ids.extend(gen.tolist())
        if gen.numel() == 0 or gen[-1].item() == tokenizer.eos_token_id or gen.numel() < budget:
            break
        seq = out
        attn = torch.ones_like(seq)
    return tokenizer.decode(new_ids, skip_special_tokens=True).strip()


def generate_judgement(client: OpenAI, model: str, question: str, reference_answer: str,
                       reply: str, judge_prompt: str):
    """Судья сравнивает reply с reference_answer (реальным ответом из test-сплита
    датасета) — без этого faithfulness была бы неверифицируемой (не с чем сверять).

    system-инструкция идёт ПЕРВЫМ сообщением, а весь оцениваемый материал — ОДНИМ
    user-сообщением, без роли assistant. Раньше reply вставлялся отдельным
    assistant-сообщением в конце списка (после system) — на open-weight чат-шаблонах
    (например minimax-m3 через vLLM/SGLang) это заставляло модель ПРОДОЛЖАТЬ чужую
    реплику вместо генерации новой оценки (сообщение с ролью assistant в конце истории
    воспринимается как недописанный ход, а не как данные для анализа), из-за чего судья
    возвращал очередной ответ на исходный вопрос вместо JSON-оценки."""
    judgement_generation = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'system',
                'content': judge_prompt
            },
            {
                'role': 'user',
                'content': (
                    f"Вопрос:\n{question}\n\n"
                    f"Эталонный ответ:\n{reference_answer}\n\n"
                    f"Ответ тестируемого:\n{reply}"
                )
            },
        ],
        # extra_body={
        #     "chat_template_kwargs": {
        #         "enable_thinking": False
        #     },
        #     "guided_json": Judgement.model_json_schema()
        # }
    )

    return judgement_generation.choices[0].message.content


def parse_judge_json(raw: str) -> dict:
    """Некоторые бэкенды (особенно внутренние, без честной поддержки
    response_format=json_object) всё равно оборачивают JSON в ```json...```
    или добавляют текст вокруг. Вырезаем содержимое между первой '{' и
    последней '}' перед парсингом; если и это не JSON — не роняем весь
    прогон, возвращаем нулевые оценки с сырым текстом в reasoning."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        print(f"== ПРЕДУПРЕЖДЕНИЕ: судья вернул невалидный JSON ({exc}), ставлю нулевые оценки")
        return {
            "faithfulness_score": 0,
            "completeness_score": 0,
            "consciousness_score": 0,
            "reasoning": f"[невалидный JSON от судьи] {raw[:2000]}",
        }


def main():
    args = parse_args()
    device = pick_device()
    print(f"Устройство: {device}")
    print(f"== judge backend: {args.openai_base_url or 'https://api.openai.com/v1 (по умолчанию)'} "
          f"| judge_model={args.judge_model}")
    client = OpenAI(api_key=args.openai_api_key, base_url=args.openai_base_url or None)

    system_prompt = args.system or DOMAIN_SYSTEM_PROMPTS[args.domain]
    judge_prompt = DOMAIN_JUDGE[args.domain]

    dataset_path = DOMAIN_DATASETS[args.domain]
    split = load_train_eval_dataset(dataset_path)[args.eval_on]
    if args.shuffle:
        split = split.shuffle(seed=args.seed)
    if args.iterations is not None:
        split = split.select(range(min(args.iterations, len(split))))
    print(f"== вопросы и эталонные ответы берутся из {args.eval_on}-сплита {dataset_path} "
          f"({len(split)} примеров)")

    metric_suffix = "" if args.eval_on == "test" else f"_{args.eval_on}"

    reply_model, reply_tokenizer = load_model(
        use_lora=args.lora,
        model_path=args.model,
        adapter_path=args.adapter,
        device=device
        )

    if args.lora:
        run_meta = load_run_meta(args.adapter)
        resume_mode = "must"
        if not run_meta:
            print(f"== {os.path.join(args.adapter, 'trackio_run.json')} не найден — "
                  "пропускаю логирование в Trackio (адаптер обучен без finetune.py?)")
    else:
        # Базовая модель без адаптера — детерминированное имя run'а по
        # модели+домену (base_run_name), а не None: иначе оценка базовой
        # модели вообще никогда не попадала на дашборд и её не с чем было
        # сравнивать. resume="allow" — первый прогон создаёт run сам,
        # следующие резюмируются в тот же.
        run_meta = {"project": TRACKIO_PROJECT, "run_name": base_run_name(args.model, args.domain)}
        resume_mode = "allow"

    if run_meta:
        trackio.init(project=run_meta["project"], name=run_meta["run_name"], resume=resume_mode)
        print(f"== метрики judge будут дописаны в Trackio-run '{run_meta['run_name']}'")

    result_dir = f"judgements/judge_{slug(args.judge_model)}"
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, args.model.rstrip('/').split('/')[-1])
    if args.lora:
        result_file += f"&&{args.adapter.rstrip('/').split('/')[-1]}"
    if args.eval_on != "test":
        result_file += f"_{args.eval_on}"
    result_file += '.txt'

    print('===Начало цикла llm-as-judge\n')
    iteration = 0
    faithfulness_scores: list[float] = []
    completeness_scores: list[float] = []
    consciousness_scores: list[float] = []
    reply_times: list[float] = []
    judge_times: list[float] = []
    entries: list[str] = []  # готовые блоки по каждому примеру -> пишем в файл одним махом
    try:
        for example in split:
            iteration += 1
            question = build_user_content(example)
            reference_answer = example["answer"]
            print(f'\n--- Пример {iteration}/{len(split)} ---')
            print(f'=Вопрос:\n{question}')

            reply_start = time.perf_counter()
            reply = generate_reply(
                model=reply_model,
                tokenizer=reply_tokenizer,
                question=question,
                system_prompt=system_prompt,
                device=device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                greedy=args.greedy)
            reply_time = time.perf_counter() - reply_start
            print(f'=Ответ модели ({reply_time:.1f} с):\n{reply}')

            judge_start = time.perf_counter()
            judge = generate_judgement(client, args.judge_model, question, reference_answer,
                                       reply, judge_prompt)
            judge_time = time.perf_counter() - judge_start
            judge_dict = parse_judge_json(judge)
            reasoning = judge_dict.get("reasoning", "Описание отсутствует")
            faithfulness = judge_dict.get("faithfulness_score", 0)
            completeness = judge_dict.get("completeness_score", 0)
            consciousness = judge_dict.get("consciousness_score", 0)
            print(f'=Оценка судьи ({judge_time:.1f} с):\n{json.dumps(judge_dict,indent=2,ensure_ascii=False)}')

            faithfulness_scores.append(faithfulness)
            completeness_scores.append(completeness)
            consciousness_scores.append(consciousness)
            reply_times.append(reply_time)
            judge_times.append(judge_time)

            if run_meta:
                # Построчные оценки — как менялась оценка от примера к примеру
                # внутри ЭТОГО прогона (не путать с judge_*_avg ниже — та метрика
                # для сравнения МЕЖДУ моделями/run'ами, эта — для просмотра
                # разброса ВНУТРИ одного прогона).
                trackio.log({
                    f"judge_faithfulness{metric_suffix}": faithfulness,
                    f"judge_completeness{metric_suffix}": completeness,
                    f"judge_consciousness{metric_suffix}": consciousness,
                }, step=iteration)

            entries.append(
                '\n==============' +
                f'\n\n==Question\n{question}' +
                f'\n\n==Reference answer\n{reference_answer}' +
                f'\n\n==Reply ({reply_time:.1f} с)\n{reply}' +
                f'\n\n==Judgement ({judge_time:.1f} с)\n' +
                f'- faithfulness = {faithfulness}\n' +
                f'- completeness = {completeness}\n' +
                f'- consciousness = {consciousness}\n' +
                f'- Reasoning\n{reasoning}\n'
            )

            if args.iterations is None:
                is_continue = input('\n====Продолжаем? (y/n): ')
                if is_continue != 'y':
                    break
    finally:
        n = len(faithfulness_scores)
        if n:
            avg_faithfulness = sum(faithfulness_scores) / n
            avg_completeness = sum(completeness_scores) / n
            avg_consciousness = sum(consciousness_scores) / n

            summary = (
                f"=== СРЕДНИЕ ОЦЕНКИ (n={n} примеров) ===\n"
                f"- faithfulness  = {avg_faithfulness:.2f}\n"
                f"- completeness  = {avg_completeness:.2f}\n"
                f"- consciousness = {avg_consciousness:.2f}\n"
            )
            print(f'\n{summary}')

            # Сводка пишется ДО всех отдельных оценок этого прогона.
            with open(result_file, 'a', encoding='utf-8') as f:
                f.write('\n' + '#' * 60 + '\n')
                f.write(summary)
                f.write('#' * 60 + '\n')
                f.write(''.join(entries))

            # _full — когда n >= FULL_EVAL_THRESHOLD (тест-сплиты обычно в разы
            # больше — требовать буквально ВСЕ непрактично), чтобы полновесная
            # проверка была отдельной метрикой, сравнимой между моделями/
            # адаптерами, а не смешивалась на графике с быстрыми выборочными
            # проверками на нескольких примерах.
            full_suffix = "_full" if n >= FULL_EVAL_THRESHOLD else ""
            avg_key = f"judge_faithfulness_avg{metric_suffix}{full_suffix}"
            n_key = f"judge_num_samples{metric_suffix}{full_suffix}"
            if run_meta and should_log_trackio_avg(run_meta["project"], run_meta["run_name"], n,
                                                   avg_key, n_key):
                # step=0 ЖЁСТКО, а не "трекио сам разберётся": Run.log() без
                # явного step берёт self._next_step, который при resume="must"
                # продолжается с максимума, уже сохранённого в БД для этого run'а
                # (см. trackio/run.py: self._next_step = 0 if max_step is None
                # else max_step + 1). Значит при повторном запуске судьи на том
                # же адаптере среднее каждый раз улетало бы на новый, всё больший
                # step — и дашборд рисовал бы через несколько запусков линию
                # вместо одного столбика. Фиксируя step=0 + should_log_trackio_avg
                # (удаляет старую строку среднего, только если у неё МЕНЬШЕ
                # примеров, чем у текущего прогона), на графике всегда остаётся
                # ровно одна, самая статистически значимая точка на run.
                trackio.log({
                    avg_key: round(avg_faithfulness, 3),
                    f"judge_completeness_avg{metric_suffix}{full_suffix}": round(avg_completeness, 3),
                    f"judge_consciousness_avg{metric_suffix}{full_suffix}": round(avg_consciousness, 3),
                    n_key: n,
                    f"judge_reply_time_avg_sec{metric_suffix}{full_suffix}": round(sum(reply_times) / n, 2),
                    f"judge_eval_time_avg_sec{metric_suffix}{full_suffix}": round(sum(judge_times) / n, 2),
                }, step=0)

        if run_meta:
            if os.path.isfile(result_file):
                # log_artifact/use_artifact появились в trackio только в июле
                # 2026 — версия, зафиксированная в проекте (0.20.2, из-за
                # требования huggingface-hub<1.0 у transformers==4.56.2), их
                # не знает вовсе. trackio.save() — доступный в этой версии
                # эквивалент: копирует файл, привязанный к ТЕКУЩЕМУ активному
                # run'у (нет name=/type=, но репорт так же попадает в файлы run'а).
                trackio.save(result_file)
                print(f"== репорт судьи сохранён как artifact в Trackio: {result_file}")
            trackio.finish()

if __name__=='__main__':
    main()
