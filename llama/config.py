from dataclasses import dataclass
import torch

@dataclass
class LlamaConfig:
    vocab_size: int
    context_length: int
    embedding_dim: int
    num_heads: int
    num_layers: int
    hidden_dim: int
    num_kv_groups: int
    rope_base: float
    dtype: torch.dtype = torch.bfloat16