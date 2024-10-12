from torch import nn
import torch

class SiLU(nn.Module):
    def forward(self, input_tensor):
        return input_tensor * torch.sigmoid(input_tensor)

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear2 = nn.Linear(input_dim, hidden_dim, bias=False)
        self.activation = SiLU()

    def forward(self, input_tensor):
        activated_output = self.activation(self.linear1(input_tensor))
        return activated_output * self.linear2(input_tensor)
