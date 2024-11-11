from torch.utils.data import Dataset
from datasets import load_dataset
from llama.util import tokenize_text
from tqdm import tqdm
import os

class TextDataset(Dataset):
    def __init__(self, data_dir, split='train', context_length=1024, tokenizer_name='gpt2', transform=None):
        self.context_length = context_length
        self.tokenizer_name = tokenizer_name
        self.transform = transform
        self.samples = []
        data_files = {split: os.path.join(data_dir, f"{split}.jsonl")}
        dataset = load_dataset('json', data_files=data_files, split=split)
        
        for sample in tqdm(dataset, desc=f"Loading {split} dataset", unit="samples"):
            text = sample['text']
            tokenized_text = tokenize_text(text, tokenizer_name).squeeze(0)
            for i in range(0, len(tokenized_text), self.context_length):
                tokens = tokenized_text[i:i + self.context_length]
                self.samples.append(tokens)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        tokens = self.samples[idx]
        if self.transform:
            tokens = self.transform(tokens)
        return tokens