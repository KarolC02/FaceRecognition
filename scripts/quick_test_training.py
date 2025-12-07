#!/usr/bin/env python3
"""
Quick test to verify training pipeline works (runs 1 epoch).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from training.config import TrainingConfig
from training.train import train_epoch, validate
from src.data.lfw_dataset import LFWDataset, get_default_transforms
from src.data.triplet_sampler import BalancedTripletSampler
from src.models.face_net import create_face_net
from src.models.losses import TripletLoss
from training.train_utils import MetricsLogger
from torch.utils.data import DataLoader
import torch.optim as optim

def main():
    print("=" * 60)
    print("Quick Training Pipeline Test")
    print("=" * 60)
    
    # Create minimal config for testing
    config = TrainingConfig()
    config.batch_size = 4
    config.num_epochs = 1
    config.num_workers = 0  # Faster for testing
    config.pretrained = False  # Faster for testing
    config.use_tensorboard = False
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create datasets
    print("\n1. Loading datasets...")
    train_transform = get_default_transforms(is_training=True)
    val_transform = get_default_transforms(is_training=False)
    
    train_dataset = LFWDataset(
        dataset_path=config.dataset_path,
        transform=train_transform,
        split="train"
    )
    val_dataset = LFWDataset(
        dataset_path=config.dataset_path,
        transform=val_transform,
        split="val"
    )
    
    print(f"   Train: {len(train_dataset)} samples")
    print(f"   Val: {len(val_dataset)} samples")
    
    # Create samplers and loaders
    print("\n2. Creating dataloaders...")
    train_sampler = BalancedTripletSampler(train_dataset, batch_size=config.batch_size)
    val_sampler = BalancedTripletSampler(val_dataset, batch_size=config.batch_size)
    
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, num_workers=0)
    
    # Create model
    print("\n3. Creating model...")
    model = create_face_net(embedding_size=config.embedding_size, pretrained=config.pretrained).to(device)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss and optimizer
    loss_fn = TripletLoss(margin=config.margin)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Create logger
    logger = MetricsLogger(log_dir="logs/test", use_tensorboard=False)
    
    # Test training epoch
    print("\n4. Testing training epoch...")
    train_metrics = train_epoch(model, train_loader, loss_fn, optimizer, device, 0, logger, config)
    print(f"   Training loss: {train_metrics['loss']:.4f}")
    
    # Test validation
    print("\n5. Testing validation...")
    val_metrics = validate(model, val_loader, loss_fn, device, 0, logger)
    print(f"   Validation loss: {val_metrics['loss']:.4f}")
    
    logger.close()
    
    print("\n" + "=" * 60)
    print("Training pipeline test passed!")
    print("=" * 60)
    print("\nYou can now run full training with:")
    print("  python training/train.py")

if __name__ == "__main__":
    main()

