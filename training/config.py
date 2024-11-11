from dataclasses import dataclass
from pathlib import Path

@dataclass
class TrainingConfig:
    learning_rate: float = 1e-2
    max_learning_rate: float = 1e-1
    num_epochs: int = 10
    patience: int = 3
    checkpoint_path: Path = Path('checkpoint.pth')
    dataset_path: str = "data/pes2o"
    batch_size: int = 8
    context_length: int = 1024