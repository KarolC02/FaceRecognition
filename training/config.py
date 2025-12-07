"""
Training configuration for face recognition model.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainingConfig:
    """Configuration for training face recognition model."""
    
    # Dataset
    dataset_path: str = "data/lfw/lfw-filtered"
    train_split: str = "train"
    val_split: str = "val"
    test_split: str = "test"
    
    # Model
    model_name: str = "facenet"  # "resnet18" or "facenet"
    embedding_size: int = 512  # 512 for FaceNet, 128 for ResNet18
    pretrained: bool = False  # Use pretrained weights (if available)
    pretrained_path: Optional[str] = None  # Path to pretrained FaceNet weights
    
    # Training
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    
    # Optimizer
    optimizer: str = "adam"  # "adam" or "sgd"
    
    # Learning rate scheduling
    lr_scheduler: str = "cosine"  # "cosine", "step", "plateau", or None
    lr_step_size: int = 20  # For step scheduler
    lr_gamma: float = 0.1  # For step scheduler
    lr_min: float = 1e-6  # Minimum learning rate
    
    # Loss
    loss_type: str = "triplet"  # "triplet" or "hard_triplet"
    margin: float = 0.2
    distance_metric: str = "euclidean"  # "euclidean" or "cosine"
    mining_strategy: str = "hard"  # For hard triplet loss
    
    # Data augmentation
    image_size: int = 224
    use_augmentation: bool = True
    
    # Training utilities
    num_workers: int = 4
    pin_memory: bool = True
    device: str = "cuda"  # "cuda" or "cpu"
    gpu_id: int = 2  # Specific GPU ID to use (0, 1, 2, 3, etc.)
    
    # Checkpointing
    save_dir: str = "data/models"
    checkpoint_freq: int = 5  # Save checkpoint every N epochs
    save_best: bool = True
    resume_from: Optional[str] = None  # Path to checkpoint to resume from
    
    # Logging
    log_dir: str = "logs"
    use_tensorboard: bool = True
    log_freq: int = 10  # Log every N batches
    
    # Validation
    val_freq: int = 1  # Validate every N epochs
    early_stopping_patience: int = 10  # Stop if no improvement for N epochs
    early_stopping_min_delta: float = 0.001
    
    # Seed
    seed: int = 42
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.optimizer in ["adam", "sgd"], f"Unknown optimizer: {self.optimizer}"
        assert self.lr_scheduler in ["cosine", "step", "plateau", None], \
            f"Unknown scheduler: {self.lr_scheduler}"
        assert self.loss_type in ["triplet", "hard_triplet"], \
            f"Unknown loss type: {self.loss_type}"
        assert self.distance_metric in ["euclidean", "cosine"], \
            f"Unknown distance metric: {self.distance_metric}"
        
        # Create directories
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def save(self, path: str):
        """Save configuration to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str):
        """Load configuration from file."""
        import json
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)

