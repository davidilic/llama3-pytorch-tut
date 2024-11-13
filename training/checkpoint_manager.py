import torch

class CheckpointManager:
    def __init__(self, config):
        self.config = config
        self.best_val_loss = None
        self.patience_counter = 0

    def save_checkpoint(self, model, optimizer, epoch, checkpoint_name: str):
        """Save model checkpoint."""
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, self.config.checkpoints_folder / checkpoint_name)
        print(f"Checkpoint saved at epoch {epoch + 1}")

    def load_checkpoint(self, model, optimizer, checkpoint_name: str):
        """Load checkpoint if it exists. Returns starting epoch."""
        checkpoint_path = self.config.checkpoints_folder / checkpoint_name
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.best_val_loss = checkpoint['best_val_loss']
            starting_epoch = checkpoint['epoch'] + 1
            print(f"Loaded checkpoint from epoch {starting_epoch}")
            return starting_epoch
        return 0

    def check_improvement(self, val_loss):
        """Check if validation loss has improved."""
        if self.best_val_loss is None or val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
            return True
        else:
            self.patience_counter += 1
            print(f"No improvement in validation loss for {self.patience_counter} epoch(s)")
            return False

    def should_stop(self):
        """Determine if training should stop early."""
        return self.patience_counter >= self.config.patience
    