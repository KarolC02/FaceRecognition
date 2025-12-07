"""
Training utilities: metrics, logging, checkpointing.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Optional
import json
from datetime import datetime


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class MetricsLogger:
    """Logs training metrics."""
    
    def __init__(self, log_dir: str, use_tensorboard: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.use_tensorboard = use_tensorboard
        self.metrics_history = []
        
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except ImportError:
                print("Warning: TensorBoard not available, disabling tensorboard logging")
                self.use_tensorboard = False
                self.writer = None
        else:
            self.writer = None
    
    def log(self, step: int, metrics: Dict[str, float], phase: str = "train"):
        """Log metrics."""
        # Store in history
        entry = {
            'step': step,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(entry)
        
        # Log to tensorboard
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{phase}/{key}', value, step)
    
    def log_epoch(self, epoch: int, metrics: Dict[str, float], phase: str = "train"):
        """Log epoch-level metrics."""
        if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(f'{phase}_epoch/{key}', value, epoch)
    
    def save_metrics(self, path: Optional[str] = None):
        """Save metrics history to JSON."""
        if path is None:
            path = self.log_dir / "metrics.json"
        with open(path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def close(self):
        """Close logger."""
        if self.writer is not None:
            self.writer.close()


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict[str, float],
    config: object,
    filepath: str,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Training metrics
        config: Training configuration
        filepath: Path to save checkpoint
        is_best: Whether this is the best model so far
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics,
        'config': config.to_dict() if hasattr(config, 'to_dict') else config,
    }
    
    # Ensure filepath is a string
    filepath = str(filepath)
    
    # Save checkpoint with error handling
    try:
        torch.save(checkpoint, filepath)
    except Exception as e:
        print(f"Error saving checkpoint: {e}")
        # Try saving without optimizer state (sometimes optimizer state can cause issues)
        checkpoint_no_opt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'config': config.to_dict() if hasattr(config, 'to_dict') else config,
        }
        torch.save(checkpoint_no_opt, filepath)
        print(f"Saved checkpoint without optimizer state to {filepath}")
    
    # Save best model separately with model-specific name
    if is_best:
        # Get model name from config
        model_name = "unknown"
        if hasattr(config, 'model_name'):
            model_name = config.model_name
        elif isinstance(config, dict):
            model_name = config.get('model_name', 'unknown')
        
        best_path = Path(filepath).parent / f"best_model_{model_name}.pth"
        torch.save(checkpoint, best_path)
        print(f"Saved best model to {best_path}")


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None
):
    """
    Load model checkpoint.
    
    Args:
        filepath: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state into
    
    Returns:
        Dictionary with checkpoint info (epoch, loss, metrics, config)
    """
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'loss': checkpoint.get('loss', float('inf')),
        'metrics': checkpoint.get('metrics', {}),
        'config': checkpoint.get('config', {}),
    }


def get_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer."""
    return optimizer.param_groups[0]['lr']


def set_learning_rate(optimizer: torch.optim.Optimizer, lr: float):
    """Set learning rate for optimizer."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' (lower is better) or 'max' (higher is better)
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current score to evaluate
        
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop
    
    def _is_better(self, current: float, best: float) -> bool:
        """Check if current score is better than best."""
        if self.mode == 'min':
            return current < best - self.min_delta
        else:
            return current > best + self.min_delta
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False

