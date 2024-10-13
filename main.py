from util import get_llama_config, setup_model, tokenize_text, generate_and_decode

def main():
    config = get_llama_config()
    model = setup_model(config)

    initial_text = "My name is"
    initial_tokens = tokenize_text(initial_text)

    print("\nInput text:", initial_text)

    generated_text = generate_and_decode(
        model=model,
        initial_tokens=initial_tokens,
        max_new_tokens=10,
        context_size=config.context_length
    )

    print("Generated text:", generated_text)

if __name__ == "__main__":
    main()