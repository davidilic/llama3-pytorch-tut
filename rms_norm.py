from torch import nn
import torch

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        means = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(means + self.eps) * self.weight