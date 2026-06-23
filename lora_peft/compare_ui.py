"""LMArena-стайл сравнение: базовая модель vs база+LoRA-адаптер, бок о бок.

Один вопрос -> два ответа в двух колонках:
  слева  — базовая модель (адаптер выключен),
  справа — та же модель с LoRA-адаптером.

Базовая модель грузится в память ОДИН раз; LoRA включается/выключается через
PeftModel.disable_adapter() — поэтому не держим в памяти две копии 4B.

Запуск из корня репозитория:
    python lora_peft/compare_ui.py
Откроется локальный веб-интерфейс (адрес напечатается в консоли, обычно
http://127.0.0.1:7860).
"""
import contextlib
import os
import sys
import threading

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from peft import PeftModel
from sft_lora_peft import MODEL_DIR, pick_device, torch_dtype

ADAPTER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lora-adapter")

# Защитный потолок на суммарную длину ответа (догенерация чанками до EOS не зациклится).
SAFETY_MAX_NEW_TOKENS = 4096

SYSTEM = ("Ты — эксперт по внутреннему аудиту и управлению рисками. "
          "Отвечай в профессиональном регистре, используя корректную "
          "терминологию, структурируя рассуждение в логике системы "
          "управления рисками. Отвечай на русском.")

device = pick_device()
print(f"== device: {device}")
print("== загрузка базовой модели (один раз)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch_dtype, low_cpu_mem_usage=True)
model = PeftModel.from_pretrained(base, ADAPTER_DIR)
model.to(device)
model.eval()
print("== модель + LoRA-адаптер готовы")


def _build_inputs(question: str):
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)


def _stream_one(question: str, use_adapter: bool, max_new_tokens: int):
    """Стримит ПОЛНЫЙ ответ одной из версий модели: генерирует чанками по
    max_new_tokens до естественной остановки (EOS) либо до защитного потолка —
    ответ не обрезается."""
    inputs = _build_inputs(question)
    seq = inputs["input_ids"]
    attn = inputs["attention_mask"]
    acc = ""
    total = 0

    while total < SAFETY_MAX_NEW_TOKENS:
        budget = min(max_new_tokens, SAFETY_MAX_NEW_TOKENS - total)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        holder: dict = {}
        cur_seq, cur_attn = seq, attn

        def run():
            with torch.no_grad():
                # выключаем LoRA для базовой колонки; для адаптера — без изменений
                ctx = contextlib.nullcontext() if use_adapter else model.disable_adapter()
                with ctx:
                    holder["out"] = model.generate(
                        input_ids=cur_seq,
                        attention_mask=cur_attn,
                        max_new_tokens=budget,
                        do_sample=False,            # greedy -> воспроизводимое сравнение
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        streamer=streamer,
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
        # закончили: пусто, EOS, или выдано меньше бюджета -> модель остановилась сама
        if gen.numel() == 0 or gen[-1].item() == tokenizer.eos_token_id or gen.numel() < budget:
            break
        # упёрлись в бюджет — продолжаем с уже сгенерированного
        seq = out
        attn = torch.ones_like(seq)


def respond(message, hist_base, hist_adapter, max_new_tokens):
    message = (message or "").strip()
    if not message:
        yield hist_base, hist_adapter, ""
        return

    # добавляем вопрос и пустые ответы в обе колонки
    hist_base = hist_base + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "_генерирую..._"}]
    hist_adapter = hist_adapter + [{"role": "user", "content": message},
                                   {"role": "assistant", "content": "_ожидание..._"}]
    yield hist_base, hist_adapter, ""

    # сначала стримим базовую модель (слева)
    for partial in _stream_one(message, use_adapter=False, max_new_tokens=max_new_tokens):
        hist_base[-1]["content"] = partial
        yield hist_base, hist_adapter, ""

    # затем — модель с адаптером (справа)
    for partial in _stream_one(message, use_adapter=True, max_new_tokens=max_new_tokens):
        hist_adapter[-1]["content"] = partial
        yield hist_base, hist_adapter, ""


with gr.Blocks(title="Сравнение моделей: base vs LoRA") as demo:
    gr.Markdown(
        "## 🆚 Сравнение ответов: базовая модель vs LoRA-адаптер\n"
        "Один вопрос — два ответа. Слева — исходная модель, справа — дообученная (LoRA). "
        "Генерация на MPS небыстрая: ответы появляются постепенно."
    )
    with gr.Row():
        cb_base = gr.Chatbot(label="⬅️ Базовая модель (без адаптера)", height=480)
        cb_adapter = gr.Chatbot(label="➡️ С LoRA-адаптером", height=480)
    msg = gr.Textbox(placeholder="Введите вопрос по внутреннему аудиту…", label="Вопрос", lines=2)
    with gr.Row():
        max_tok = gr.Slider(128, 2048, value=1024, step=64,
                            label="Размер чанка генерации (ответ догенерируется до конца)")
        send = gr.Button("Отправить", variant="primary")
        clear = gr.Button("Очистить")

    inp = [msg, cb_base, cb_adapter, max_tok]
    out = [cb_base, cb_adapter, msg]
    send.click(respond, inp, out)
    msg.submit(respond, inp, out)
    clear.click(lambda: ([], [], ""), None, out)


if __name__ == "__main__":
    demo.launch()
