#!/usr/bin/env python3
"""
Test script to verify the dataset loader and model work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.lfw_dataset import LFWDataset, get_default_transforms
from src.models.face_net import create_face_net
import torch


def test_dataset():
    """Test the LFW dataset loader."""
    print("=" * 60)
    print("Testing LFW Dataset Loader")
    print("=" * 60)
    
    dataset_path = "data/lfw/lfw-filtered"  # Use filtered dataset
    
    # Test train dataset
    print("\n1. Creating train dataset...")
    train_transform = get_default_transforms(is_training=True)
    train_dataset = LFWDataset(
        dataset_path=dataset_path,
        transform=train_transform,
        split="train"
    )
    
    print(f"   Train dataset size: {len(train_dataset)}")
    print(f"   Number of classes: {train_dataset.get_num_classes()}")
    
    # Test val dataset
    print("\n2. Creating validation dataset...")
    val_transform = get_default_transforms(is_training=False)
    val_dataset = LFWDataset(
        dataset_path=dataset_path,
        transform=val_transform,
        split="val"
    )
    
    print(f"   Val dataset size: {len(val_dataset)}")
    print(f"   Number of classes: {val_dataset.get_num_classes()}")
    
    # Test getting a sample
    print("\n3. Testing sample retrieval...")
    if len(train_dataset) > 0:
        image, label = train_dataset[0]
        print(f"   Sample image shape: {image.shape}")
        print(f"   Sample label: {label}")
        print(f"   Person name: {train_dataset.id_to_person[label]}")
    
    # Test DataLoader
    print("\n4. Testing DataLoader...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0  # Set to 0 for testing, increase for actual training
    )
    
    batch_images, batch_labels = next(iter(train_loader))
    print(f"   Batch images shape: {batch_images.shape}")
    print(f"   Batch labels shape: {batch_labels.shape}")
    print(f"   Unique labels in batch: {len(torch.unique(batch_labels))}")
    
    return train_dataset, val_dataset


def test_model():
    """Test the FaceNet model."""
    print("\n" + "=" * 60)
    print("Testing FaceNet Model")
    print("=" * 60)
    
    # Create model
    print("\n1. Creating model...")
    model = create_face_net(embedding_size=128, pretrained=True)
    print(f"   Model created with embedding size: {model.embedding_size}")
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    model.eval()
    with torch.no_grad():
        embedding = model(dummy_input)
    
    print(f"   Input shape: {dummy_input.shape}")
    print(f"   Embedding shape: {embedding.shape}")
    print(f"   Embedding norm (should be ~1.0): {embedding.norm(dim=1).mean().item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n3. Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    return model


def test_end_to_end():
    """Test dataset + model together."""
    print("\n" + "=" * 60)
    print("Testing End-to-End (Dataset + Model)")
    print("=" * 60)
    
    # Create dataset
    transform = get_default_transforms(is_training=False)
    dataset = LFWDataset(
        dataset_path="data/lfw/lfw-dataset",
        transform=transform,
        split="train"
    )
    
    # Create model
    model = create_face_net(embedding_size=128, pretrained=True)
    model.eval()
    
    # Get a batch
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0
    )
    
    batch_images, batch_labels = next(iter(dataloader))
    
    # Forward pass
    with torch.no_grad():
        embeddings = model(batch_images)
    
    print(f"\nBatch processed successfully!")
    print(f"   Images shape: {batch_images.shape}")
    print(f"   Labels: {batch_labels.tolist()}")
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embedding norms: {embeddings.norm(dim=1).tolist()}")


if __name__ == "__main__":
    try:
        # Test dataset
        train_dataset, val_dataset = test_dataset()
        
        # Test model
        model = test_model()
        
        # Test end-to-end
        test_end_to_end()
        
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

