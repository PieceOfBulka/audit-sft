from lora_class import LoRALayer
import torch.nn as nn

class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(in_features=self.linear.in_features,
                              out_features=self.linear.out_features,
                              rank=rank,
                              alpha=alpha)
        
        for param in self.linear.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        return self.linear(x) + self.lora(x)