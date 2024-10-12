import torch
from torch import nn
from rms_norm import RMSNorm
from block import TransformerLayer
from config import LlamaConfig

class Llama3(nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embedding_dim, dtype=config.dtype)

        self.trf_blocks = nn.Sequential(
            *[TransformerLayer(config) for _ in range(config.num_layers)]
        )

        self.final_norm = RMSNorm(config.embedding_dim, eps=1e-5)
        self.out_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False, dtype=config.dtype)

    def forward(self, in_idx):
        tok_embeds = self.tok_emb(in_idx)
        x = tok_embeds
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x.to(torch.float32))
        return logits