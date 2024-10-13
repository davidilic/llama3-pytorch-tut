from typing import Any, Dict, Protocol
import torch
from torch import nn

class RoPEParams:
    @staticmethod
    def precompute(head_dim, theta_base=10_000, context_length=4096):
        inv_freq = 1.0 / (theta_base ** (torch.arange(0, head_dim // 2) / (head_dim // 2)))
        positions = torch.arange(context_length)
        angles = positions[:, None] * inv_freq[None, :]
        angles = torch.cat([angles, angles], dim=1)
        return torch.cos(angles), torch.sin(angles)

    @staticmethod
    def apply(x, cos, sin):
        batch_size, num_heads, seq_len, head_dim = x.shape
        x1, x2 = x[..., :head_dim//2], x[..., head_dim//2:]
        cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
        sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)
        rotated = torch.cat((-x2, x1), dim=-1)
        return (x * cos + rotated * sin).to(dtype=x.dtype)

class SharedBuffers:
    _buffers = {}

    @classmethod
    def get(cls, context_length, head_dim, rope_base, freq_config, dtype=torch.float32):
        key = (context_length, head_dim, rope_base, tuple(freq_config.values()) if freq_config else freq_config, dtype)
        if key not in cls._buffers:
            mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
            cos, sin = RoPEParams.precompute(head_dim, rope_base, context_length)
            cls._buffers[key] = (mask, cos.to(dtype), sin.to(dtype))
        return cls._buffers[key]

class AttentionConfig(Protocol):
    embedding_dim: int
    context_length: int
    num_heads: int
    num_kv_groups: int
    rope_base: int
    rope_freq: Dict[str, Any]
    dtype: Any

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0, "embedding_dim must be divisible by num_heads"
        assert config.num_heads % config.num_kv_groups == 0, "num_heads must be divisible by num_kv_groups"
        
        self.d_out = config.embedding_dim
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.num_kv_groups = config.num_kv_groups
        self.group_size = config.num_heads // config.num_kv_groups

        self.W_query = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False, dtype=config.dtype)
        self.W_key = nn.Linear(config.embedding_dim, config.num_kv_groups * self.head_dim, bias=False, dtype=config.dtype)
        self.W_value = nn.Linear(config.embedding_dim, config.num_kv_groups * self.head_dim, bias=False, dtype=config.dtype)
        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim, bias=False, dtype=config.dtype)

        mask, cos, sin = SharedBuffers.get(config.context_length, self.head_dim, config.rope_base, config.rope_freq, config.dtype)
        self.register_buffer("mask", mask)
        self.register_buffer("cos", cos)
        self.register_buffer("sin", sin)

    def forward(self, x):
        b, num_tokens, _ = x.shape
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        keys = self.W_key(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_kv_groups, self.head_dim).transpose(1, 2)

        queries = RoPEParams.apply(queries, self.cos, self.sin)
        keys = RoPEParams.apply(keys, self.cos, self.sin)

        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], float('-inf'))
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)

        context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)