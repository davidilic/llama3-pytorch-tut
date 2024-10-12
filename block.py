from torch import nn
from config import LlamaConfig
from rms_norm import RMSNorm
from feedforward import FeedForward
from attention import GroupedQueryAttention

class TransformerLayer(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        
        self.attention = GroupedQueryAttention(
            config.embedding_dim,
            config.embedding_dim,
            config.context_length,
            config.num_heads,
            config.num_kv_groups,
            config.rope_base,
            config.rope_freq,
            config.dtype
        )

        self.feed_forward = FeedForward(
            config.embedding_dim,
            config.hidden_dim,
            config.dtype
        )

        self.attention_norm = RMSNorm(config.embedding_dim)
        self.feed_forward_norm = RMSNorm(config.embedding_dim)

    def forward(self, x):
        attention_output = self.attention(self.attention_norm(x))
        attention_residual = x + attention_output
        feed_forward_output = self.feed_forward(self.feed_forward_norm(attention_residual))
        return attention_residual + feed_forward_output
