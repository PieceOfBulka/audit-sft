from linear_lora import LinearWithLoRA
import torch

def merge_lora_weights(model):
    all_modules=dict(model.named_modules())
    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            with torch.no_grad():
                merged = module.lora.B @ module.lora.A * module.lora.scaling
                module.linear.weight.data += merged.T
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            if parent_name:
                parent = all_modules[parent_name]
            else:
                parent = model
            setattr(parent, child_name, module.linear)