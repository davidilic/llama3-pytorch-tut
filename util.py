import tiktoken
import torch
from llama3 import Llama3, LlamaConfig

def generate_text(model, initial_tokens, max_new_tokens, context_size):
    generated_tokens = initial_tokens
    for _ in range(max_new_tokens):
        input_tokens = generated_tokens[:, -context_size:]
        with torch.no_grad():
            logits = model(input_tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
    return generated_tokens

def get_llama_config():
    return LlamaConfig(
        vocab_size=50257,
        context_length=512,
        embedding_dim=1024,
        num_heads=8,
        num_layers=6,
        hidden_dim=4096,
        num_kv_groups=4,
        rope_base=10000,
        rope_freq={
            "factor": 32,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_context_length": 512
        },
        dtype=torch.bfloat16
    )

def setup_model(config):
    torch.manual_seed(11)
    model = Llama3(config)
    model.eval()
    return model

def tokenize_text(text, tokenizer_name="gpt2"):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)

def generate_and_decode(model, initial_tokens, max_new_tokens, context_size, tokenizer_name="gpt2"):
    generated_tokens = generate_text(
        model=model,
        initial_tokens=initial_tokens,
        max_new_tokens=max_new_tokens,
        context_size=context_size
    )
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    return tokenizer.decode(generated_tokens.squeeze(0).tolist())