from torch import nn
import torch

class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dtype=None):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False, dtype=dtype)
        self.linear2 = nn.Linear(input_dim, hidden_dim, bias=False, dtype=dtype)
        self.activation = SiLU()

    def forward(self, x):
        activated_output = self.activation(self.linear1(x))
        return activated_output * self.linear2(x)
