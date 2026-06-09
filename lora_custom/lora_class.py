import torch.nn as nn
import torch
import math

class LoRALayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: int = 16):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha/rank

        self.B = nn.Parameter(torch.zeros(rank, out_features))
        self.A = nn.Parameter(torch.randn(in_features,rank) * 1/math.sqrt(rank))

    def forward(self, x):
        return self.B @ self.A @ x * self.scaling