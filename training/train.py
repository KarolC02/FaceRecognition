#!/usr/bin/env python3
"""
Main training script for face recognition model.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import argparse

from src.data.lfw_dataset import LFWDataset, get_default_transforms
from src.data.triplet_sampler import BalancedTripletSampler
from src.models.face_net import create_face_net
from src.models.facenet_pretrained import create_facenet_pretrained
from src.models.losses import TripletLoss, HardTripletLoss
from training.config import TrainingConfig
from training.train_utils import (
    AverageMeter, MetricsLogger, save_checkpoint, load_checkpoint,
    EarlyStopping, get_learning_rate
)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: MetricsLogger,
    config: TrainingConfig
):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter('Loss', ':.4f')
    d_ap_meter = AverageMeter('d_ap', ':.4f')
    d_an_meter = AverageMeter('d_an', ':.4f')
    violation_meter = AverageMeter('Violations', ':.2%')
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, batch in enumerate(pbar):
        # DataLoader with batch_sampler returns [images_tensor, labels_tensor]
        # batch[0] = images tensor [batch_size*3, 3, 224, 224]
        # batch[1] = labels tensor [batch_size*3]
        images = batch[0].to(device)
        labels = batch[1].to(device)
        
        # Forward pass
        embeddings = model(images)
        
        # Split into triplets
        anchors = embeddings[0::3]
        positives = embeddings[1::3]
        negatives = embeddings[2::3]
        
        # Compute loss
        loss, metrics = loss_fn(anchors, positives, negatives)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update meters
        loss_meter.update(loss.item(), len(anchors))
        d_ap_meter.update(metrics['d_ap_mean'], len(anchors))
        d_an_meter.update(metrics['d_an_mean'], len(anchors))
        violation_meter.update(metrics['margin_violations'], len(anchors))
        
        # Logging
        if batch_idx % config.log_freq == 0:
            step = epoch * len(dataloader) + batch_idx
            logger.log(step, {
                'loss': loss_meter.avg,
                'd_ap': d_ap_meter.avg,
                'd_an': d_an_meter.avg,
                'violations': violation_meter.avg,
                'lr': get_learning_rate(optimizer)
            }, phase='train')
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'd_ap': f'{d_ap_meter.avg:.4f}',
            'd_an': f'{d_an_meter.avg:.4f}',
            'viol': f'{violation_meter.avg:.2%}'
        })
    
    return {
        'loss': loss_meter.avg,
        'd_ap': d_ap_meter.avg,
        'd_an': d_an_meter.avg,
        'violations': violation_meter.avg
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    epoch: int,
    logger: MetricsLogger
):
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter('Loss', ':.4f')
    d_ap_meter = AverageMeter('d_ap', ':.4f')
    d_an_meter = AverageMeter('d_an', ':.4f')
    violation_meter = AverageMeter('Violations', ':.2%')
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
        
        for batch in pbar:
            # DataLoader with batch_sampler returns [images_tensor, labels_tensor]
            # batch[0] = images tensor [batch_size*3, 3, 224, 224]
            # batch[1] = labels tensor [batch_size*3]
            images = batch[0].to(device)
            labels = batch[1].to(device)
            
            # Forward pass
            embeddings = model(images)
            
            # Split into triplets
            anchors = embeddings[0::3]
            positives = embeddings[1::3]
            negatives = embeddings[2::3]
            
            # Compute loss
            loss, metrics = loss_fn(anchors, positives, negatives)
            
            # Update meters
            loss_meter.update(loss.item(), len(anchors))
            d_ap_meter.update(metrics['d_ap_mean'], len(anchors))
            d_an_meter.update(metrics['d_an_mean'], len(anchors))
            violation_meter.update(metrics['margin_violations'], len(anchors))
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss_meter.avg:.4f}',
                'd_ap': f'{d_ap_meter.avg:.4f}',
                'd_an': f'{d_an_meter.avg:.4f}',
                'viol': f'{violation_meter.avg:.2%}'
            })
    
    # Log epoch metrics
    logger.log_epoch(epoch, {
        'loss': loss_meter.avg,
        'd_ap': d_ap_meter.avg,
        'd_an': d_an_meter.avg,
        'violations': violation_meter.avg
    }, phase='val')
    
    return {
        'loss': loss_meter.avg,
        'd_ap': d_ap_meter.avg,
        'd_an': d_an_meter.avg,
        'violations': violation_meter.avg
    }


def main():
    parser = argparse.ArgumentParser(description='Train face recognition model')
    parser.add_argument('--config', type=str, help='Path to config file (optional)')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        choices=['resnet18', 'facenet'],
        help='Model architecture: resnet18 or facenet (overrides config if provided)'
    )
    parser.add_argument(
        '--embedding-size',
        type=int,
        default=None,
        help='Embedding size (default: 512 for facenet, 128 for resnet18)'
    )
    parser.add_argument(
        '--pretrained-path',
        type=str,
        default=None,
        help='Path to pretrained weights (optional)'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=None,
        help='Number of epochs to train (overrides config)'
    )
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = TrainingConfig.load(args.config)
    else:
        config = TrainingConfig()
    
    # Update config based on args (only if explicitly provided)
    if args.model:
        config.model_name = args.model
    
    # Set embedding size based on model if not specified
    if args.embedding_size:
        config.embedding_size = args.embedding_size
    elif args.model == 'facenet':
        config.embedding_size = 512
    else:  # resnet18
        config.embedding_size = 128
    
    if args.pretrained_path:
        config.pretrained_path = args.pretrained_path
        config.pretrained = True
    
    if args.num_epochs:
        config.num_epochs = args.num_epochs
        print(f"Training for {config.num_epochs} epochs (overridden from command line)")
    
    print(f"Training with model: {config.model_name}")
    print(f"Embedding size: {config.embedding_size}")
    if config.pretrained_path:
        print(f"Pretrained path: {config.pretrained_path}")
    print()
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    
    # Device
    if torch.cuda.is_available() and config.device == "cuda":
        # Set specific GPU
        torch.cuda.set_device(config.gpu_id)
        device = torch.device(f"cuda:{config.gpu_id}")
        print(f"Using GPU {config.gpu_id}: {torch.cuda.get_device_name(config.gpu_id)}")
    else:
        device = torch.device('cpu')
        print(f"Using device: {device}")
    
    # Create datasets
    print("Loading datasets...")
    train_transform = get_default_transforms(
        image_size=config.image_size,
        is_training=config.use_augmentation
    )
    val_transform = get_default_transforms(
        image_size=config.image_size,
        is_training=False
    )
    
    train_dataset = LFWDataset(
        dataset_path=config.dataset_path,
        transform=train_transform,
        split=config.train_split
    )
    val_dataset = LFWDataset(
        dataset_path=config.dataset_path,
        transform=val_transform,
        split=config.val_split
    )
    
    print(f"Train dataset: {len(train_dataset)} samples, {train_dataset.get_num_classes()} people")
    print(f"Val dataset: {len(val_dataset)} samples, {val_dataset.get_num_classes()} people")
    
    # Create samplers
    train_sampler = BalancedTripletSampler(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        seed=config.seed
    )
    val_sampler = BalancedTripletSampler(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        seed=config.seed
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Create model
    print("Creating model...")
    if config.model_name == "facenet":
        print(f"Using FaceNet architecture (embedding_size={config.embedding_size})")
        model = create_facenet_pretrained(
            embedding_size=config.embedding_size,
            pretrained=config.pretrained,
            pretrained_path=config.pretrained_path,
        ).to(device)
    else:  # resnet18
        print(f"Using ResNet18 architecture (embedding_size={config.embedding_size})")
        model = create_face_net(
            embedding_size=config.embedding_size,
            pretrained=config.pretrained
        ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create loss
    if config.loss_type == "triplet":
        loss_fn = TripletLoss(
            margin=config.margin,
            distance_metric=config.distance_metric
        )
    else:  # hard_triplet
        loss_fn = HardTripletLoss(
            margin=config.margin,
            distance_metric=config.distance_metric,
            mining_strategy=config.mining_strategy
        )
    
    # Create optimizer
    if config.optimizer == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )
    
    # Create learning rate scheduler
    if config.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.num_epochs,
            eta_min=config.lr_min
        )
    elif config.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.lr_step_size,
            gamma=config.lr_gamma
        )
    elif config.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=config.lr_min
        )
    else:
        scheduler = None
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    if args.resume or config.resume_from:
        checkpoint_path = args.resume or config.resume_from
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = load_checkpoint(checkpoint_path, model, optimizer)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['loss']
        print(f"Resumed from epoch {checkpoint['epoch']}, loss: {best_val_loss:.4f}")
    
    # Create logger
    logger = MetricsLogger(
        log_dir=config.log_dir,
        use_tensorboard=config.use_tensorboard
    )
    
    # Save config
    config.save(Path(config.save_dir) / "config.json")
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.early_stopping_min_delta,
        mode='min'
    )
    
    # Training loop
    print("\nStarting training...")
    print("=" * 60)
    
    final_epoch = start_epoch - 1
    for epoch in range(start_epoch, config.num_epochs):
        final_epoch = epoch
        # Train
        train_metrics = train_epoch(
            model, train_loader, loss_fn, optimizer, device,
            epoch, logger, config
        )
        
        # Validate
        if epoch % config.val_freq == 0:
            val_metrics = validate(
                model, val_loader, loss_fn, device, epoch, logger
            )
            
            # Update learning rate
            if scheduler is not None:
                if config.lr_scheduler == "plateau":
                    scheduler.step(val_metrics['loss'])
                else:
                    scheduler.step()
            
            # Track best model (but don't save yet)
            is_best = val_metrics['loss'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['loss']
                best_epoch = epoch
                # Save best model state for final checkpoint (copy to CPU to save memory)
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                print(f"  New best validation loss: {best_val_loss:.4f} (epoch {epoch})")
            
            # Early stopping
            if early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping at epoch {epoch}")
                print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
                break
        
        print()
    
    # Save final checkpoint (best model)
    if final_epoch >= start_epoch and best_model_state is not None:
        # Load best model state back to device
        best_model_state_device = {k: v.to(device) for k, v in best_model_state.items()}
        model.load_state_dict(best_model_state_device)
        
        # Determine checkpoint name based on whether we're fine-tuning
        if config.pretrained and config.pretrained_path:
            checkpoint_name = f"checkpoint_{config.model_name}_finetuned_final.pth"
            note = 'Final checkpoint - fine-tuned from pretrained'
        else:
            checkpoint_name = f"checkpoint_{config.model_name}_final.pth"
            note = 'Final checkpoint - trained from scratch'
        
        final_checkpoint_path = Path(config.save_dir) / checkpoint_name
        final_metrics = {'loss': best_val_loss, 'epoch': best_epoch, 'note': note}
        save_checkpoint(
            model, optimizer, best_epoch, best_val_loss,
            final_metrics, config, str(final_checkpoint_path), is_best=False
        )
        print(f"Saved final checkpoint (best model from epoch {best_epoch}) to {final_checkpoint_path}")
    
    # Save final metrics
    logger.save_metrics()
    logger.close()
    
    print("=" * 60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")
    if final_epoch >= start_epoch and best_model_state is not None:
        if config.pretrained and config.pretrained_path:
            print(f"Final checkpoint saved at: {Path(config.save_dir) / f'checkpoint_{config.model_name}_finetuned_final.pth'}")
        else:
            print(f"Final checkpoint saved at: {Path(config.save_dir) / f'checkpoint_{config.model_name}_final.pth'}")


if __name__ == "__main__":
    main()

