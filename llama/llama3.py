import torch
from torch import nn
from llama.rms_norm import RMSNorm
from llama.block import TransformerBlock
from llama.config import LlamaConfig

class Llama3(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.token_embedding = nn.Embedding(config.vocab_size, config.embedding_dim, dtype=config.dtype)
        self.transformer_layers = nn.Sequential(*[TransformerBlock(config) for _ in range(config.num_layers)])
        self.layer_norm = RMSNorm(config.embedding_dim)
        self.output_projection = nn.Linear(config.embedding_dim, config.vocab_size, bias=False, dtype=config.dtype)
        self.llama_model = nn.Sequential(self.token_embedding, self.transformer_layers, self.layer_norm, self.output_projection)

    def forward(self, input_token_ids):
        return self.llama_model(input_token_ids)