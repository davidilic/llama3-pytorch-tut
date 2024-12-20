from training.llama_trainer import LlamaTrainer
from training.config import TrainingConfig
from llama.config import LlamaConfig
from llama.llama3 import Llama3
from pathlib import Path
import torch

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    llama_config = LlamaConfig(
        vocab_size=50257,
        context_length=1024,
        embedding_dim=512,
        num_heads=8,
        num_layers=8,
        hidden_dim=2048,
        num_kv_groups=4,
        rope_base=10000,
        dtype=torch.bfloat16
    )

    training_config = TrainingConfig(
        learning_rate=1e-4,
        max_learning_rate=3e-4,
        num_epochs=3,
        patience=2,
        checkpoints_folder=Path('checkpoints/'),
        dataset_path="data/pes2o",
        batch_size=16,
        context_length=1024
    )
    
    model = Llama3(llama_config).to(device)
    trainer = LlamaTrainer(model, training_config, device)
    trainer.train()

if __name__ == "__main__":
    main()
