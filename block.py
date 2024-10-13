from typing import Any, Dict, Protocol
from torch import nn
from rms_norm import RMSNorm
from feedforward import FeedForward
from attention import GroupedQueryAttention

class TransformerBlockConfig(Protocol):
    embedding_dim: int
    context_length: int
    num_heads: int
    num_kv_groups: int
    rope_base: int
    rope_freq: Dict[str, Any]
    dtype: Any
    hidden_dim: int


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerBlockConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config.embedding_dim, config.hidden_dim, config.dtype)
        self.attention_norm = RMSNorm(config.embedding_dim)
        self.feed_forward_norm = RMSNorm(config.embedding_dim)

    def forward(self, x):
        attention_output = self.attention(self.attention_norm(x))
        attention_residual = x + attention_output
        feed_forward_output = self.feed_forward(self.feed_forward_norm(attention_residual))
        return attention_residual + feed_forward_output
