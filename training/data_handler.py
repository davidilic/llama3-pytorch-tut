from torch.utils.data import DataLoader
from training.config import TrainingConfig
from data.dataset import TextDataset
from tqdm import tqdm
import torch

class DataHandler:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.train_dataloader = None
        self.val_dataloader = None 
        self.test_dataloader = None
        self._prepare_dataloaders()

    def _prepare_dataloaders(self):
        """Initialize train, validation, and test dataloaders."""
        def collate_fn(batch):
            return torch.nn.utils.rnn.pad_sequence(batch, batch_first=True)
    
        splits = ['train', 'validation', 'test']
        datasets = {
            split: TextDataset(
                data_dir=self.config.dataset_path,
                split=split,
                context_length=self.config.context_length,
                tokenizer_name='gpt2'
            ) for split in splits
        }
    
        dataloaders = {
            split: DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=(split == 'train'),
                collate_fn=collate_fn
            ) for split, dataset in tqdm(datasets.items(), desc="Preparing dataloaders", unit="split")
        }
        
        self.train_dataloader = dataloaders['train']
        self.val_dataloader = dataloaders['validation'] 
        self.test_dataloader = dataloaders['test']

    def get_dataloaders(self):
        """Return the dataloaders."""
        return self.train_dataloader, self.val_dataloader, self.test_dataloader
    
    def get_train_dataloader(self):
        """Return the training dataloader."""
        return self.train_dataloader
    
    def get_val_dataloader(self):
        """Return the validation dataloader."""
        return self.val_dataloader
    
    def get_test_dataloader(self):
        """Return the test dataloader."""
        return self.test_dataloader
    