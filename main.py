import tiktoken
import torch
from config import LlamaConfig
from llama3 import Llama3

def generate_text(model, initial_tokens, max_new_tokens, context_size):
    generated_tokens = initial_tokens
    for _ in range(max_new_tokens):
        input_tokens = generated_tokens[:, -context_size:]
        with torch.no_grad():
            logits = model(input_tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
    return generated_tokens

def main():

    llama_config = LlamaConfig(
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
        dtype=torch.float32
    )

    torch.manual_seed(11)
    model = Llama3(llama_config)
    model.eval()

    initial_text = "Test input"
    tokenizer = tiktoken.get_encoding("gpt2") # using gpt2 tokenizer for convenience
    initial_tokens = torch.tensor(tokenizer.encode(initial_text)).unsqueeze(0)

    print("\nInput text:", initial_text)

    generated_tokens = generate_text(
        model=model,
        initial_tokens=initial_tokens,
        max_new_tokens=10,
        context_size=llama_config.context_length
    )

    generated_text = tokenizer.decode(generated_tokens.squeeze(0).tolist())
    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()