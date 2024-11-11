import tiktoken
import torch

def generate_text(model, initial_tokens, max_new_tokens, context_size):
    generated_tokens = initial_tokens
    for _ in range(max_new_tokens):
        input_tokens = generated_tokens[:, -context_size:]
        with torch.no_grad():
            logits = model(input_tokens)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
    return generated_tokens

def tokenize_text(text, tokenizer_name="gpt2"):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    return torch.tensor(tokenizer.encode(text)).unsqueeze(0)

def decode_text(tokens, tokenizer_name="gpt2"):
    tokenizer = tiktoken.get_encoding(tokenizer_name)
    return tokenizer.decode(tokens.squeeze(0).tolist())

def generate_and_decode(model, initial_tokens, max_new_tokens, context_size, tokenizer_name="gpt2"):
    generated_tokens = generate_text(model, initial_tokens, max_new_tokens, context_size)
    return decode_text(generated_tokens, tokenizer_name)