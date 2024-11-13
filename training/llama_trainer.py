from dataclasses import dataclass
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler
from dataclasses import dataclass
from training.data_handler import DataHandler
from training.checkpoint_manager import CheckpointManager
from tqdm import tqdm

@dataclass
class MetricsLog:
    epoch: int
    train_loss: float
    train_accuracy: float
    train_perplexity: float
    val_loss: float
    val_accuracy: float
    val_perplexity: float

class LlamaTrainer:
    def __init__(self, model, config, device):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)

        self.data_handler = DataHandler(config)
        self.checkpoint_manager = CheckpointManager(config)

        self.writer = SummaryWriter()
        total_steps = config.num_epochs * len(self.data_handler.get_train_dataloader())
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.max_learning_rate,
            total_steps=total_steps,
            anneal_strategy='linear'
        )

    def train(self):
        """Main training loop."""
        self.model.train()
        train_dataloader, val_dataloader, test_dataloader = self.data_handler.get_dataloaders()

        starting_epoch = self.checkpoint_manager.load_checkpoint(self.model, self.optimizer, "llama_latest.pth") or \
                        self.checkpoint_manager.load_checkpoint(self.model, self.optimizer, "llama_best.pth")
        
        print("Starting epoch: ", starting_epoch)

        for epoch in tqdm(range(starting_epoch, self.config.num_epochs), desc="Epochs", unit="epoch"):
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            train_loss, train_acc, train_ppl = self.train_epoch(train_dataloader, epoch)
            print("Train Loss: ", train_loss, "Train Accuracy: ", train_acc, "Train Perplexity: ", train_ppl)

            val_loss, val_acc = self.evaluate(val_dataloader)
            val_ppl = torch.exp(torch.tensor(val_loss)).item()

            metrics_log = MetricsLog(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                train_perplexity=train_ppl,
                val_loss=val_loss,
                val_accuracy=val_acc,
                val_perplexity=val_ppl
            )

            self._log_metrics(metrics_log)

            if self.checkpoint_manager.check_improvement(val_loss):
                self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, epoch, "llama_best.pth")

            if self.checkpoint_manager.should_stop():
                print("Early stopping triggered.")
                break

        self._final_evaluation(test_dataloader)

    def train_epoch(self, train_dataloader, epoch):
        """Train for one epoch and return metrics."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc="Training", unit="batch")):
            batch = batch.to(self.device)
            self.optimizer.zero_grad()

            logits = self.model(batch[:, :-1])
            targets = batch[:, 1:]
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )
            
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item() * batch.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.numel()

            global_step = epoch * len(train_dataloader) + batch_idx
            self.writer.add_scalar('Train/Loss_batch', loss.item(), global_step)

            if batch_idx % 5000 == 0:
                self.checkpoint_manager.save_checkpoint(self.model, self.optimizer, epoch, "llama_latest.pth")

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        perplexity = torch.exp(torch.tensor(avg_loss))

        return avg_loss, accuracy, perplexity.item()

    def _log_metrics(self, metrics):
        """Log metrics to TensorBoard."""
        self.writer.add_scalar('Train/Loss_epoch', metrics.train_loss, metrics.epoch)
        self.writer.add_scalar('Train/Accuracy_epoch', metrics.train_accuracy, metrics.epoch)
        self.writer.add_scalar('Train/Perplexity_epoch', metrics.train_perplexity, metrics.epoch)

        self.writer.add_scalar('Validation/Loss', metrics.val_loss, metrics.epoch)
        self.writer.add_scalar('Validation/Accuracy', metrics.val_accuracy, metrics.epoch)
        self.writer.add_scalar('Validation/Perplexity', metrics.val_perplexity, metrics.epoch)

    @torch.no_grad()
    def evaluate(self, dataloader):
        """Evaluate the model and return metrics."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            batch = batch.to(self.device)
            logits = self.model(batch[:, :-1])
            targets = batch[:, 1:]

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            total_loss += loss.item() * batch.size(0)
            preds = logits.argmax(dim=-1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.numel()

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples

        return avg_loss, accuracy

    def _final_evaluation(self, test_dataloader):
        """Perform final evaluation on test set."""
        if self.config.checkpoint_path.exists():
            checkpoint = torch.load(self.config.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded best model for evaluation.")

        test_loss, test_acc = self.evaluate(test_dataloader)
        test_ppl = torch.exp(torch.tensor(test_loss)).item()
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}, Test Perplexity: {test_ppl:.4f}")