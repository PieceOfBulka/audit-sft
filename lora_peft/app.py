#!/usr/bin/env python3
"""Единый Gradio-интерфейс: чат/сравнение моделей + запуск обучения + Trackio.

Три вкладки:
  1. Чат / Сравнение — выбор модели+адаптера, обычный чат или split-screen
     сравнение (база vs дообученная).
  2. Обучение — запуск lora_peft/finetune.py с выбором модели, LoRA-параметров
     и датасета (существующего или только что загруженного).
  3. Trackio — встроенный дашборд экспериментов (обучение + bertscore + judge).

Запуск из корня репозитория:
    python lora_peft/app.py
"""
import contextlib
import glob
import json
import os
import subprocess
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from peft import PeftModel

from lora_peft.common import (DOMAIN_DATASETS, DOMAIN_SYSTEM_PROMPTS,
                               TRACKIO_PROJECT, list_available_adapters,
                               list_available_models, pick_device,
                               resolve_model_dir, silence_max_length_warning)

# Без этого OPENAI_TOKEN/INTERNAL_API_TOKEN из .env не попадают в os.environ
# этого процесса, и launch_judge() ниже всегда читал бы дефолт "not-needed" —
# .env читается только дочерним llm_as_judge.py, а ключ для CLI-аргумента
# --openai-api-key нужен уже здесь, в родительском процессе.
load_dotenv()

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEVICE = pick_device()
NO_ADAPTER = "— без адаптера —"
SAFETY_MAX_NEW_TOKENS = 4096
DEFAULT_SYSTEM_PROMPT = DOMAIN_SYSTEM_PROMPTS.get("audit", "Отвечай на русском.")
CUSTOM_PROMPT_LABEL = "— свой промпт —"


def _torch_dtype_for(device: str):
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


# =====================================================================
# Вкладка 1: Чат / Сравнение
# =====================================================================

class LoadedModel:
    """Хранит текущую загруженную модель, чтобы не грузить веса заново на
    каждый запрос. Загружает ОДИН раз базовую модель, LoRA-адаптер
    включается/выключается через PeftModel.disable_adapter(), либо не
    навешивается вовсе, если адаптер не выбран."""

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_id = None
        self.adapter_path = None
        self.has_adapter = False

    def ensure(self, model_id: str, adapter_path: str | None):
        adapter_path = adapter_path if adapter_path and adapter_path != NO_ADAPTER else None
        if self.model is not None and self.model_id == model_id and self.adapter_path == adapter_path:
            return  # уже загружено то, что нужно

        if self.model is not None:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        model_dir = resolve_model_dir(model_id)
        dtype = _torch_dtype_for(DEVICE)

        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype, low_cpu_mem_usage=True)
        self.has_adapter = adapter_path is not None
        if self.has_adapter:
            model = PeftModel.from_pretrained(model, adapter_path)
        model.to(DEVICE)
        model.eval()
        silence_max_length_warning(model)

        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.adapter_path = adapter_path

    def status(self) -> str:
        if self.model is None:
            return "Модель не загружена."
        adapter_note = f" + LoRA: {self.adapter_path}" if self.has_adapter else " (без адаптера)"
        return f"Загружено: {self.model_id}{adapter_note} | device={DEVICE}"


LOADED = LoadedModel()


def refresh_choices():
    models = list_available_models()
    adapters = [NO_ADAPTER] + list_available_adapters()
    return (gr.update(choices=models, value=models[0] if models else None),
            gr.update(choices=adapters, value=NO_ADAPTER))


def load_model_ui(model_id, adapter_path):
    if not model_id:
        return "Сначала выбери модель (кнопка 'Обновить списки', если пусто)."
    try:
        LOADED.ensure(model_id, adapter_path)
    except Exception as exc:  # noqa: BLE001 — показываем причину пользователю в UI
        return f"Ошибка загрузки: {exc}"
    return LOADED.status()


def _build_inputs(tokenizer, system_prompt: str, question: str):
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(DEVICE)


def _stream_one(model, tokenizer, system_prompt: str, question: str, use_adapter: bool,
                max_new_tokens: int):
    inputs = _build_inputs(tokenizer, system_prompt, question)
    seq, attn = inputs["input_ids"], inputs["attention_mask"]
    acc, total = "", 0

    while total < SAFETY_MAX_NEW_TOKENS:
        budget = min(max_new_tokens, SAFETY_MAX_NEW_TOKENS - total)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        holder: dict = {}
        cur_seq, cur_attn = seq, attn

        def run():
            with torch.no_grad():
                is_peft = isinstance(model, PeftModel)
                ctx = contextlib.nullcontext() if (use_adapter or not is_peft) else model.disable_adapter()
                with ctx:
                    holder["out"] = model.generate(
                        input_ids=cur_seq, attention_mask=cur_attn, max_new_tokens=budget,
                        do_sample=False, pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id, streamer=streamer,
                    )

        t = threading.Thread(target=run, daemon=True)
        t.start()
        for chunk in streamer:
            acc += chunk
            yield acc
        t.join()

        out = holder.get("out")
        if out is None:
            break
        gen = out[0][cur_seq.shape[1]:]
        total += int(gen.numel())
        if gen.numel() == 0 or gen[-1].item() == tokenizer.eos_token_id or gen.numel() < budget:
            break
        seq, attn = out, torch.ones_like(out)


def chat_respond(message, history, system_prompt, max_new_tokens):
    message = (message or "").strip()
    if not message or LOADED.model is None:
        yield history, ""
        return
    history = history + [{"role": "user", "content": message},
                         {"role": "assistant", "content": "_генерирую..._"}]
    yield history, ""
    for partial in _stream_one(LOADED.model, LOADED.tokenizer, system_prompt, message,
                               use_adapter=True, max_new_tokens=max_new_tokens):
        history[-1]["content"] = partial
        yield history, ""


def compare_respond(message, hist_base, hist_adapter, system_prompt, max_new_tokens):
    message = (message or "").strip()
    if not message or LOADED.model is None:
        yield hist_base, hist_adapter, ""
        return
    hist_base = hist_base + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "_генерирую..._"}]
    hist_adapter = hist_adapter + [{"role": "user", "content": message},
                                   {"role": "assistant", "content": "_ожидание..._"}]
    yield hist_base, hist_adapter, ""

    for partial in _stream_one(LOADED.model, LOADED.tokenizer, system_prompt, message,
                               use_adapter=False, max_new_tokens=max_new_tokens):
        hist_base[-1]["content"] = partial
        yield hist_base, hist_adapter, ""

    for partial in _stream_one(LOADED.model, LOADED.tokenizer, system_prompt, message,
                               use_adapter=True, max_new_tokens=max_new_tokens):
        hist_adapter[-1]["content"] = partial
        yield hist_base, hist_adapter, ""


def build_chat_tab():
    with gr.Tab("💬 Чат / Сравнение"):
        gr.Markdown("Выбери модель и (опционально) LoRA-адаптер, затем нажми **Загрузить модель**.")
        with gr.Row():
            model_dd = gr.Dropdown(choices=list_available_models(), label="Модель (weights/<repo-id>)")
            adapter_dd = gr.Dropdown(choices=[NO_ADAPTER] + list_available_adapters(),
                                     value=NO_ADAPTER, label="LoRA-адаптер")
            refresh_btn = gr.Button("🔄 Обновить списки")
        with gr.Row():
            load_btn = gr.Button("📥 Загрузить модель", variant="primary")
            status_box = gr.Textbox(label="Статус", value="Модель не загружена.", interactive=False)

        with gr.Row():
            system_preset_dd = gr.Dropdown(choices=[CUSTOM_PROMPT_LABEL] + sorted(DOMAIN_SYSTEM_PROMPTS),
                                           value="audit" if "audit" in DOMAIN_SYSTEM_PROMPTS else CUSTOM_PROMPT_LABEL,
                                           label="Системный промпт: пресет")
        system_box = gr.Textbox(label="Системный промпт (можно править после выбора пресета)",
                                value=DEFAULT_SYSTEM_PROMPT, lines=3)
        compare_mode = gr.Checkbox(label="Режим сравнения (база vs адаптер, split-screen)", value=False)

        with gr.Row(visible=False) as compare_row:
            cb_base = gr.Chatbot(label="⬅️ Базовая модель (без адаптера)", height=460)
            cb_adapter = gr.Chatbot(label="➡️ С LoRA-адаптером", height=460)
        cb_single = gr.Chatbot(label="Чат", height=460)

        msg = gr.Textbox(placeholder="Введите вопрос…", label="Вопрос", lines=2)
        with gr.Row():
            max_tok = gr.Slider(128, 2048, value=1024, step=64, label="Размер чанка генерации")
            send = gr.Button("Отправить", variant="primary")
            clear = gr.Button("Очистить")

        refresh_btn.click(refresh_choices, None, [model_dd, adapter_dd])
        load_btn.click(load_model_ui, [model_dd, adapter_dd], status_box)
        compare_mode.change(lambda v: (gr.update(visible=v), gr.update(visible=not v)),
                            compare_mode, [compare_row, cb_single])

        def apply_preset(preset):
            if preset == CUSTOM_PROMPT_LABEL:
                return gr.update()  # оставить текущий текст как есть — свой промпт
            return gr.update(value=DOMAIN_SYSTEM_PROMPTS[preset])

        system_preset_dd.change(apply_preset, system_preset_dd, system_box)

        def route_send(message, h_single, h_base, h_adapter, system_prompt, max_new_tokens, compare):
            if compare:
                for h_b, h_a, m in compare_respond(message, h_base, h_adapter, system_prompt, max_new_tokens):
                    yield h_single, h_b, h_a, m
            else:
                for h, m in chat_respond(message, h_single, system_prompt, max_new_tokens):
                    yield h, h_base, h_adapter, m

        send_inputs = [msg, cb_single, cb_base, cb_adapter, system_box, max_tok, compare_mode]
        send_outputs = [cb_single, cb_base, cb_adapter, msg]
        send.click(route_send, send_inputs, send_outputs)
        msg.submit(route_send, send_inputs, send_outputs)
        clear.click(lambda: ([], [], [], ""), None, send_outputs)


# =====================================================================
# Вкладка 2: Обучение
# =====================================================================

ALL_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_proc_runner():
    """Возвращает (run, stop) пару для запуска CLI-скрипта как подпроцесса со
    стримингом stdout в лог-бокс. Отдельный holder на каждый вызов — так
    вкладки 'Обучение' и 'Оценка' не мешают друг другу отдельными кнопками
    'Остановить'."""
    holder: dict[str, subprocess.Popen | None] = {"proc": None}

    def run(cmd: list[str]):
        if holder["proc"] is not None and holder["proc"].poll() is None:
            yield "Процесс уже запущен — дождись завершения или останови его."
            return
        log = f"$ {' '.join(cmd)}\n\n"
        yield log
        proc = subprocess.Popen(cmd, cwd=_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        holder["proc"] = proc
        for line in proc.stdout:
            log += line
            yield log
        proc.wait()
        log += f"\n== процесс завершён с кодом {proc.returncode}\n"
        yield log

    def stop():
        proc = holder["proc"]
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return "Остановлено (SIGTERM отправлен)."
        return "Активного процесса нет."

    return run, stop


def _dataset_choice_names():
    return sorted(DOMAIN_DATASETS)


_run_training, stop_training = make_proc_runner()


def launch_training(model_id, dataset_choice, uploaded_file, custom_system_prompt,
                    adapter_name, rank, alpha, dropout, target_modules,
                    lr, batch_size, grad_accum, epochs, warmup_ratio,
                    group_by_length, liger, load_best):
    cmd = [sys.executable, os.path.join(_ROOT, "lora_peft", "finetune.py"),
           "--model", model_id,
           "--rank", str(rank), "--alpha", str(alpha), "--lora-dropout", str(dropout),
           "--target-modules", ",".join(target_modules) if target_modules else ",".join(ALL_TARGET_MODULES),
           "--lr", str(lr), "--batch-size", str(batch_size),
           "--grad-accum-steps", str(grad_accum), "--epochs", str(epochs),
           "--warmup-ratio", str(warmup_ratio)]
    if adapter_name:
        cmd += ["--adapter-name", adapter_name]
    if not group_by_length:
        cmd += ["--no-group-by-length"]
    if not liger:
        cmd += ["--no-liger"]
    if load_best:
        cmd += ["--load-best-model-at-end"]

    if uploaded_file is not None:
        if not custom_system_prompt:
            yield "Для нового датасета нужно задать системный промпт."
            return
        upload_dir = os.path.join(_ROOT, "data", "uploaded")
        os.makedirs(upload_dir, exist_ok=True)
        dest = os.path.join(upload_dir, os.path.basename(uploaded_file))
        with open(uploaded_file, "rb") as src, open(dest, "wb") as dst:
            dst.write(src.read())
        cmd += ["--dataset", dest, "--system-prompt", custom_system_prompt]
    else:
        if not dataset_choice:
            yield "Выбери существующий датасет или загрузи новый."
            return
        cmd += ["--domain", dataset_choice]

    yield from _run_training(cmd)


def build_train_tab():
    with gr.Tab("🏋️ Обучение"):
        gr.Markdown("Запуск `lora_peft/finetune.py` с выбранными параметрами. "
                    "Лог обучения появится ниже по мере выполнения.")
        with gr.Row():
            model_dd = gr.Dropdown(choices=list_available_models(), allow_custom_value=True,
                                   label="Модель (repo-id или weights/<путь>)")
            refresh_models_btn = gr.Button("🔄")

        with gr.Row():
            dataset_dd = gr.Dropdown(choices=_dataset_choice_names(), label="Существующий датасет (--domain)")
            upload_file = gr.File(label="…или загрузить новый датасет (.json)", file_types=[".json"])
        custom_system = gr.Textbox(label="Системный промпт (обязателен для нового датасета)",
                                   lines=2, visible=True)

        adapter_name = gr.Textbox(label="Имя адаптера (папка ./lora-adapter/<имя>)")

        with gr.Row():
            rank = gr.Slider(4, 64, value=16, step=4, label="rank")
            alpha = gr.Slider(8, 128, value=32, step=8, label="alpha")
            dropout = gr.Slider(0.0, 0.3, value=0.0, step=0.01,
                                label="lora_dropout (0 = быстрый fused-путь Unsloth)")
        target_modules = gr.CheckboxGroup(choices=ALL_TARGET_MODULES, value=ALL_TARGET_MODULES,
                                          label="target_modules")
        with gr.Row():
            lr = gr.Number(value=1e-4, label="learning_rate")
            batch_size = gr.Slider(1, 64, value=8, step=1, label="per_device_batch_size")
            grad_accum = gr.Slider(1, 16, value=1, step=1, label="gradient_accumulation_steps")
            epochs = gr.Slider(1, 10, value=2, step=1, label="epochs")
        with gr.Row():
            warmup_ratio = gr.Slider(0.0, 0.2, value=0.03, step=0.01, label="warmup_ratio")
            group_by_length = gr.Checkbox(value=True, label="group_by_length")
            liger = gr.Checkbox(value=True, label="use_liger_kernel")
            load_best = gr.Checkbox(value=False, label="load_best_model_at_end")

        with gr.Row():
            start_btn = gr.Button("▶️ Запустить обучение", variant="primary")
            stop_btn = gr.Button("⏹ Остановить")
        log_box = gr.Textbox(label="Лог", lines=25, max_lines=25, interactive=False, autoscroll=True)

        refresh_models_btn.click(lambda: gr.update(choices=list_available_models()), None, model_dd)

        start_inputs = [model_dd, dataset_dd, upload_file, custom_system, adapter_name,
                        rank, alpha, dropout, target_modules, lr, batch_size, grad_accum,
                        epochs, warmup_ratio, group_by_length, liger, load_best]
        start_btn.click(launch_training, start_inputs, log_box)
        stop_btn.click(stop_training, None, log_box)


# =====================================================================
# Вкладка 3: Оценка (BERTScore + LLM-as-judge)
# =====================================================================

# Пресеты OpenAI-совместимого бэкенда для судьи. api_key_env — переменная в
# .env, откуда берётся ключ (не хардкодим сами значения в код).
JUDGE_BACKENDS = {
    "OpenAI (gpt-5-nano)": {
        "base_url": None,  # None -> official OpenAI, ничего не передаём
        "model": "gpt-5-nano",
        "api_key_env": "OPENAI_TOKEN",
    },
    "Внутренний сервер (minimax-m3)": {
        "base_url": "http://10.246.6.82:8080/v1",
        "model": "minimax-m3",
        "api_key_env": "INTERNAL_API_TOKEN",
    },
}

_run_bertscore, stop_bertscore = make_proc_runner()
_run_judge, stop_judge = make_proc_runner()


def launch_bertscore(model_id, adapter_path, dataset_choice, base_only, num, batch_size, eval_on):
    cmd = [sys.executable, os.path.join(_ROOT, "lora_peft", "bertscore.py"),
           "--model", model_id, "--domain", dataset_choice,
           "--num", str(num), "--batch-size", str(batch_size), "--eval-on", eval_on]
    if base_only or not adapter_path or adapter_path == NO_ADAPTER:
        cmd += ["--base-only"]
    else:
        cmd += ["--adapter", adapter_path]
    yield from _run_bertscore(cmd)


def launch_judge(model_id, adapter_path, dataset_choice, base_only, backend_name, iterations, greedy, eval_on):
    backend = JUDGE_BACKENDS[backend_name]
    api_key = os.environ.get(backend["api_key_env"], "not-needed")

    # Без --shuffle: всегда одни и те же первые N примеров выбранного сплита
    # (порядок самих сплитов уже детерминирован фиксированным seed в
    # load_dataset.py), чтобы разные модели/адаптеры можно было честно
    # сравнивать по одинаковым данным.
    cmd = [sys.executable, os.path.join(_ROOT, "lora_peft", "llm_as_judge.py"),
           "--model", resolve_model_dir(model_id), "--domain", dataset_choice,
           "--judge-model", backend["model"], "--eval-on", eval_on,
           "--openai-api-key", api_key, "--iterations", str(iterations)]
    if backend["base_url"]:
        cmd += ["--openai-base-url", backend["base_url"]]
    if not base_only and adapter_path and adapter_path != NO_ADAPTER:
        cmd += ["--lora", "--adapter", adapter_path]
    if greedy:
        cmd += ["--greedy"]
    yield from _run_judge(cmd)


def build_eval_tab():
    with gr.Tab("📈 Оценка"):
        gr.Markdown("Оценка модели/адаптера через `bertscore.py` (метрика к эталонным ответам) "
                    "и `llm_as_judge.py` (LLM-судья по сгенерированным вопросам). "
                    "Оба пишут метрики в тот же Trackio-run, что и обучение (если есть trackio_run.json рядом с адаптером).")

        with gr.Row():
            eval_model_dd = gr.Dropdown(choices=list_available_models(), allow_custom_value=True,
                                        label="Модель")
            eval_adapter_dd = gr.Dropdown(choices=[NO_ADAPTER] + list_available_adapters(),
                                          value=NO_ADAPTER, label="LoRA-адаптер")
            eval_refresh_btn = gr.Button("🔄")
        eval_dataset_dd = gr.Dropdown(choices=_dataset_choice_names(), label="Датасет (--domain)")
        eval_base_only = gr.Checkbox(value=False, label="Оценивать базовую модель (без адаптера)")
        eval_on_radio = gr.Radio(choices=["test", "train"], value="test", label="Сплит",
                                 info="test — реальное качество; train — контрольный прогон "
                                      "на обучающих данных (сравнить с test при подозрении на "
                                      "переобучение или на 'нечестные' вопросы в test)")

        eval_refresh_btn.click(refresh_choices, None, [eval_model_dd, eval_adapter_dd])

        with gr.Accordion("BERTScore", open=True):
            with gr.Row():
                bs_num = gr.Slider(5, 500, value=50, step=5, label="Число примеров")
                bs_batch = gr.Slider(1, 32, value=8, step=1, label="batch_size (генерация)")
            with gr.Row():
                bs_start_btn = gr.Button("▶️ Запустить BERTScore", variant="primary")
                bs_stop_btn = gr.Button("⏹ Остановить")
            bs_log = gr.Textbox(label="Лог BERTScore", lines=15, max_lines=15,
                                interactive=False, autoscroll=True)
            bs_start_btn.click(
                launch_bertscore,
                [eval_model_dd, eval_adapter_dd, eval_dataset_dd, eval_base_only, bs_num, bs_batch,
                 eval_on_radio],
                bs_log,
            )
            bs_stop_btn.click(stop_bertscore, None, bs_log)

        with gr.Accordion("LLM-as-judge", open=True):
            gr.Markdown("Ключи бэкендов берутся из `.env`: `OPENAI_TOKEN` (официальный OpenAI) "
                        "или `INTERNAL_API_TOKEN` (внутренний сервер).")
            with gr.Row():
                judge_backend = gr.Radio(choices=list(JUDGE_BACKENDS), value="OpenAI (gpt-5-nano)",
                                         label="Бэкенд судьи/генератора вопросов")
                judge_iterations = gr.Slider(1, 50, value=5, step=1,
                                             label="Сколько примеров из test-сплита оценить")
                judge_greedy = gr.Checkbox(value=False, label="greedy-генерация ответа")
            with gr.Row():
                judge_start_btn = gr.Button("▶️ Запустить LLM-as-judge", variant="primary")
                judge_stop_btn = gr.Button("⏹ Остановить")
            judge_log = gr.Textbox(label="Лог LLM-as-judge", lines=20, max_lines=20,
                                   interactive=False, autoscroll=True)
            judge_start_btn.click(
                launch_judge,
                [eval_model_dd, eval_adapter_dd, eval_dataset_dd, eval_base_only,
                 judge_backend, judge_iterations, judge_greedy, eval_on_radio],
                judge_log,
            )
            judge_stop_btn.click(stop_judge, None, judge_log)


# =====================================================================
# Вкладка 4: Trackio
# =====================================================================

_TRACKIO_STATE: dict[str, int | None] = {"port": None}


def open_trackio_dashboard(request: gr.Request):
    import urllib.parse

    if _TRACKIO_STATE["port"] is None:
        import trackio
        # host="0.0.0.0" — иначе Trackio слушает только 127.0.0.1 и снаружи
        # (с другой машины) до него не достучаться, даже если основной
        # app.py открыт на 0.0.0.0. Возвращает (server, local_url,
        # share_url, full_url); full_url несёт write_token (право
        # писать/удалять run'ы) — для просмотра в iframe он не нужен.
        _server, local_url, _share_url, _full_url = trackio.show(
            project=TRACKIO_PROJECT, open_browser=False, block_thread=False,
            host="0.0.0.0",
        )
        _TRACKIO_STATE["port"] = urllib.parse.urlparse(local_url).port

    # Хост берём из заголовка запроса, которым реально открыли app.py в
    # браузере (например IP сервера), а не 127.0.0.1 из local_url — иначе
    # с удалённого ноутбука iframe будет стучаться в свой собственный localhost.
    browser_host = request.headers.get("host", "127.0.0.1:7860").split(":")[0]
    url = f"http://{browser_host}:{_TRACKIO_STATE['port']}/?project={TRACKIO_PROJECT}"

    html = (f'<iframe src="{url}" style="width:100%; height:80vh; border:none;"></iframe>'
           f'<p>Если дашборд не открылся во фрейме — <a href="{url}" target="_blank">открыть в новой вкладке</a>. '
           f'Если не открывается вообще — проверь, что порт {_TRACKIO_STATE["port"]} '
           f'разрешён в firewall сервера (как и {7860}).</p>')
    return html


def build_trackio_tab():
    with gr.Tab("📊 Trackio"):
        gr.Markdown("Все прогоны обучения + BERTScore + LLM-as-judge из одного проекта Trackio.")
        open_btn = gr.Button("Открыть/обновить дашборд", variant="primary")
        frame = gr.HTML()
        open_btn.click(open_trackio_dashboard, None, frame)


# =====================================================================

with gr.Blocks(title="LoRA Finetuning Studio") as demo:
    gr.Markdown("# LoRA Finetuning Studio")
    build_chat_tab()
    build_train_tab()
    build_eval_tab()
    build_trackio_tab()

demo.queue()

if __name__ == "__main__":
    # server_name="0.0.0.0" — слушать все интерфейсы, а не только localhost,
    # чтобы открывалось по http://<IP сервера>:7860 с рабочего ноутбука.
    # server_port фиксирован, чтобы firewall при желании мог разрешить
    # ровно этот один порт, а не диапазон.
    demo.launch(server_name="0.0.0.0", server_port=7860)
