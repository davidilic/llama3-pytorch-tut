from activations import SwiGLU
from torch import nn

class FeedForward(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dtype):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, embedding_dim, dtype=dtype, bias=False)
        self.swiglu = SwiGLU(embedding_dim, hidden_dim, dtype=dtype)

    def forward(self, x):
        return self.linear(self.swiglu(x))