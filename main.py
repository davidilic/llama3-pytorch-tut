from training.llama_trainer import LlamaTrainer
from training.config import TrainingConfig
from llama.config import LlamaConfig
from llama.llama3 import Llama3
from pathlib import Path
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    llama_config = LlamaConfig(
        vocab_size=50257,        # Reduced from 50257
        context_length=1024,     # Keep as is
        embedding_dim=512,       # Reduced from 624
        num_heads=8,            # Reduced from 12
        num_layers=8,           # Reduced from 10
        hidden_dim=2048,        # Reduced from 3072
        num_kv_groups=4,        # Keep as is
        rope_base=10000,        # Keep as is
        dtype=torch.bfloat16    # Keep as is
    )

    training_config = TrainingConfig(
        learning_rate=1e-4,      # Reduced from 1e-3
        max_learning_rate=3e-4,  # Reduced from 5e-4
        num_epochs=3,           # Reduced from 10
        patience=2,             # Reduced from 3
        checkpoint_path=Path('checkpoints/llama_100m.pth'),
        dataset_path="data/pes2o",
        batch_size=16,          # Reduced from 32
        context_length=1024     # Keep as is
    )
    
    model = Llama3(llama_config).to(device)
    trainer = LlamaTrainer(model, training_config, device)
    trainer.train()

if __name__ == "__main__":
    main()
