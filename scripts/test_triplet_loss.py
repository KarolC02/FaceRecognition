#!/usr/bin/env python3
"""
Test script for triplet loss and triplet sampler.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data.lfw_dataset import LFWDataset, get_default_transforms
from src.data.triplet_sampler import TripletSampler, BalancedTripletSampler
from src.models.face_net import create_face_net
from src.models.losses import TripletLoss, HardTripletLoss


def test_triplet_loss():
    """Test triplet loss with dummy data."""
    print("=" * 60)
    print("Testing Triplet Loss")
    print("=" * 60)
    
    batch_size = 8
    embedding_dim = 128
    
    # Create dummy embeddings
    anchor = torch.randn(batch_size, embedding_dim)
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_dim)  # Close to anchor
    negative = torch.randn(batch_size, embedding_dim)  # Far from anchor
    
    # Normalize embeddings (as our model does)
    anchor = torch.nn.functional.normalize(anchor, p=2, dim=1)
    positive = torch.nn.functional.normalize(positive, p=2, dim=1)
    negative = torch.nn.functional.normalize(negative, p=2, dim=1)
    
    # Test standard triplet loss
    print("\n1. Standard Triplet Loss:")
    triplet_loss = TripletLoss(margin=0.2, distance_metric="euclidean")
    loss, metrics = triplet_loss(anchor, positive, negative)
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   d(anchor, positive): {metrics['d_ap_mean']:.4f}")
    print(f"   d(anchor, negative): {metrics['d_an_mean']:.4f}")
    print(f"   Margin violations: {metrics['margin_violations']:.2%}")
    print(f"   Hard triplets: {metrics['hard_triplets']}/{batch_size}")
    
    # Test hard triplet loss
    print("\n2. Hard Triplet Loss:")
    hard_loss = HardTripletLoss(margin=0.2, mining_strategy="hard")
    loss_hard, metrics_hard = hard_loss(anchor, positive, negative)
    
    print(f"   Loss: {loss_hard.item():.4f}")
    print(f"   Hard triplets: {metrics_hard['hard_triplets']}/{metrics_hard['total_triplets']}")
    print(f"   Hard ratio: {metrics_hard['hard_ratio']:.2%}")
    
    return True


def test_triplet_sampler():
    """Test triplet sampler with LFW dataset."""
    print("\n" + "=" * 60)
    print("Testing Triplet Sampler with LFW Dataset")
    print("=" * 60)
    
    dataset_path = "data/lfw/lfw-filtered"
    
    # Create dataset
    print("\n1. Loading dataset...")
    transform = get_default_transforms(is_training=True)
    dataset = LFWDataset(
        dataset_path=dataset_path,
        transform=transform,
        split="train"
    )
    
    print(f"   Dataset size: {len(dataset)}")
    print(f"   Number of people: {dataset.get_num_classes()}")
    
    # Test TripletSampler
    print("\n2. Testing TripletSampler...")
    sampler = TripletSampler(
        dataset=dataset,
        batch_size=8,
        samples_per_person=2,
        shuffle=True
    )
    
    print(f"   Number of batches: {len(sampler)}")
    
    # Get first batch
    batch_indices = next(iter(sampler))
    print(f"   First batch indices length: {len(batch_indices)} (should be 8*3=24)")
    
    # Load batch
    batch_images = []
    batch_labels = []
    for idx in batch_indices:
        img, label = dataset[idx]
        batch_images.append(img)
        batch_labels.append(label)
    
    batch_images = torch.stack(batch_images)
    batch_labels = torch.tensor(batch_labels)
    
    print(f"   Batch images shape: {batch_images.shape}")
    print(f"   Batch labels: {batch_labels.tolist()}")
    
    # Verify triplet structure
    anchors = batch_labels[0::3]
    positives = batch_labels[1::3]
    negatives = batch_labels[2::3]
    
    print(f"\n   Triplet verification:")
    print(f"   Anchors: {anchors.tolist()}")
    print(f"   Positives: {positives.tolist()}")
    print(f"   Negatives: {negatives.tolist()}")
    
    # Check that anchors == positives
    matches = (anchors == positives).all()
    print(f"   Anchors == Positives: {matches.item()}")
    
    # Check that negatives are different
    different = (anchors != negatives).all()
    print(f"   Negatives are different: {different.item()}")
    
    return True


def test_end_to_end():
    """Test complete pipeline: dataset -> sampler -> model -> loss."""
    print("\n" + "=" * 60)
    print("Testing End-to-End Pipeline")
    print("=" * 60)
    
    dataset_path = "data/lfw/lfw-filtered"
    
    # Create dataset
    print("\n1. Setting up dataset and sampler...")
    transform = get_default_transforms(is_training=True)
    dataset = LFWDataset(
        dataset_path=dataset_path,
        transform=transform,
        split="train"
    )
    
    sampler = BalancedTripletSampler(
        dataset=dataset,
        batch_size=4,
        shuffle=True
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0
    )
    
    # Create model
    print("\n2. Creating model...")
    model = create_face_net(embedding_size=128, pretrained=False)  # Use False for faster testing
    model.eval()
    
    # Create loss
    triplet_loss = TripletLoss(margin=0.2, distance_metric="euclidean")
    
    # Get a batch
    print("\n3. Processing batch...")
    batch_indices = next(iter(sampler))
    
    # Load images
    batch_images = torch.stack([dataset[idx][0] for idx in batch_indices])
    batch_labels = torch.tensor([dataset[idx][1] for idx in batch_indices])
    
    print(f"   Batch images shape: {batch_images.shape}")
    print(f"   Batch labels: {batch_labels.tolist()}")
    
    # Forward pass
    print("\n4. Forward pass...")
    with torch.no_grad():
        embeddings = model(batch_images)
    
    print(f"   Embeddings shape: {embeddings.shape}")
    print(f"   Embedding norm: {embeddings.norm(dim=1).mean().item():.4f} (should be ~1.0)")
    
    # Split into triplets
    anchors = embeddings[0::3]
    positives = embeddings[1::3]
    negatives = embeddings[2::3]
    
    print(f"\n5. Computing triplet loss...")
    loss, metrics = triplet_loss(anchors, positives, negatives)
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   d(anchor, positive): {metrics['d_ap_mean']:.4f}")
    print(f"   d(anchor, negative): {metrics['d_an_mean']:.4f}")
    print(f"   Margin violations: {metrics['margin_violations']:.2%}")
    
    print("\nEnd-to-end test passed!")
    
    return True


if __name__ == "__main__":
    try:
        # Test triplet loss
        test_triplet_loss()
        
        # Test triplet sampler
        test_triplet_sampler()
        
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

