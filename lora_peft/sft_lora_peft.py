import os

from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import torch

def pick_device():
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'

device=pick_device()
# Корень репозитория = родитель каталога lora_peft. Путь к весам не зависит от cwd.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "weights", "Qwen", "Qwen3-4B-Instruct-2507")

if device == "cuda":
    torch_dtype = torch.bfloat16  # Идеально для современных GPU
elif device == "mps":
    torch_dtype = torch.float16   # Стандарт и стабильность для Mac
else:
    torch_dtype = torch.float32   # На CPU bfloat16/float16 работают очень медленно


def get_model_with_lora(rank: int = 16, alpha: int = 32, lora_dropout: float = 0.05, target_modules: list[str] = ["q_proj", "v_proj"]):
    print('==Start model initialisation')

    if device == 'cuda':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, 
            dtype=torch_dtype, 
            device_map='auto',  # 'auto' работает надежнее и лучше распределяет слои, чем жесткое 'cuda' (автоматически low_cpu_mem_usage=True)
        )
    elif device == 'mps':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, 
            dtype=torch_dtype, 
            device_map='mps',       # Сразу собирает модель в памяти чипа Apple Silicon
            low_cpu_mem_usage=True  # Включаем (True), чтобы Mac не завис при чтении весов с диска
        )
    else: # Вариант для CPU
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR, 
            dtype=torch_dtype, 
            low_cpu_mem_usage=True
        )

    print('==Weights were successfully initialised')

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules
        # target_modules=["q_proj", "v_proj"] # только query, value - поменяет стилистику, но модели будет тяжело глубоко изменить словарный запас и логику мышления
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    """
    модель Qwen3-4B-Instruct-2507 (4,022,468,096 параметров):
    - если брать "q_proj", "v_proj", то процент обучаемых параметров = 0.1464%
    - если брать all-linear, то 0.8145%
    """

    model = get_peft_model(model, lora_config)

    # gradient checkpointing: пересчитываем активации в backward вместо хранения
    # — резко снижает память (критично для 4B на MPS). Для PEFT обязателен
    # enable_input_require_grads(), иначе градиенты не дойдут до LoRA-слоёв.
    model.config.use_cache = False  # несовместимо с checkpointing
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    model.print_trainable_parameters()

    return model