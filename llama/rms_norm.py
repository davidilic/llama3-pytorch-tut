import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, dimension, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon
        self.scale = nn.Parameter(torch.ones(dimension))

    def forward(self, x):
        mean_squared = x.pow(2).mean(dim=-1, keepdim=True)
        normalization_factor = torch.rsqrt(mean_squared + self.epsilon)
        normalized_input = x * normalization_factor
        return (normalized_input * self.scale).to(x.dtype)