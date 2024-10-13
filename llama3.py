import torch
from torch import nn
from rms_norm import RMSNorm
from block import TransformerBlock
from typing import Optional
from dataclasses import dataclass

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
    rope_freq: Optional[dict] = None
    dtype: torch.dtype = torch.bfloat16

class Llama3(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, dtype=config.dtype)

        self.transformer_layers = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config.num_layers)]
        )

        self.layer_norm = RMSNorm(config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self, input_token_ids):
        embeddings = self.token_embedding(input_token_ids)
        x = embeddings
        x = self.transformer_layers(x)
        x = self.layer_norm(x)
        logits = self.output_projection(x)
        return logits