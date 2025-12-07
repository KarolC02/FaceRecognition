#!/usr/bin/env python3
"""
Evaluate trained models on the test set.
Computes accuracy, precision, recall, F1-score, and other metrics.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, confusion_matrix, classification_report
)
from tqdm import tqdm
import argparse
import json
from collections import defaultdict

from src.data.lfw_dataset import LFWDataset, get_default_transforms
from src.data.triplet_sampler import BalancedTripletSampler
from src.models.face_net import create_face_net
from src.models.facenet_pretrained import create_facenet_pretrained
from src.models.losses import TripletLoss
from training.train_utils import load_checkpoint
from training.config import TrainingConfig


def evaluate_triplet_loss(model, dataloader, loss_fn, device):
    """Evaluate triplet loss on test set."""
    model.eval()
    
    total_loss = 0.0
    total_d_ap = 0.0
    total_d_an = 0.0
    total_violations = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating triplet loss"):
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
            
            total_loss += loss.item()
            total_d_ap += metrics['d_ap_mean']
            total_d_an += metrics['d_an_mean']
            total_violations += metrics['margin_violations']
            num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'd_ap': total_d_ap / num_batches,
        'd_an': total_d_an / num_batches,
        'violations': total_violations / num_batches
    }


def evaluate_verification(model, test_dataset, device, num_pairs=1000, threshold=0.6):
    """
    Evaluate face verification accuracy.
    For each person, sample pairs and compute accuracy.
    """
    model.eval()
    
    # Group samples by person
    person_to_indices = defaultdict(list)
    for idx, (_, label) in enumerate(test_dataset):
        person_to_indices[label].append(idx)
    
    # Generate positive and negative pairs
    positive_pairs = []
    negative_pairs = []
    
    # Positive pairs (same person)
    for person_id, indices in person_to_indices.items():
        if len(indices) >= 2:
            # Sample pairs from same person
            for _ in range(min(5, len(indices) * (len(indices) - 1) // 2)):
                if len(positive_pairs) >= num_pairs // 2:
                    break
                idx1, idx2 = np.random.choice(indices, 2, replace=False)
                positive_pairs.append((idx1, idx2, 1))  # 1 = same person
    
    # Negative pairs (different people)
    person_ids = list(person_to_indices.keys())
    for _ in range(num_pairs - len(positive_pairs)):
        p1, p2 = np.random.choice(person_ids, 2, replace=False)
        idx1 = np.random.choice(person_to_indices[p1])
        idx2 = np.random.choice(person_to_indices[p2])
        negative_pairs.append((idx1, idx2, 0))  # 0 = different person
    
    all_pairs = positive_pairs + negative_pairs
    np.random.shuffle(all_pairs)
    
    # Compute embeddings for all images
    print("Computing embeddings...")
    embeddings_dict = {}
    transform = get_default_transforms(is_training=False)
    
    with torch.no_grad():
        for idx1, idx2, label in tqdm(all_pairs, desc="Computing embeddings"):
            if idx1 not in embeddings_dict:
                img1, _ = test_dataset[idx1]
                img1_tensor = img1.unsqueeze(0).to(device)
                emb1 = model(img1_tensor).cpu()
                embeddings_dict[idx1] = emb1
            
            if idx2 not in embeddings_dict:
                img2, _ = test_dataset[idx2]
                img2_tensor = img2.unsqueeze(0).to(device)
                emb2 = model(img2_tensor).cpu()
                embeddings_dict[idx2] = emb2
    
    # Compute distances and predictions
    print("Computing distances and predictions...")
    distances = []
    labels_true = []
    
    for idx1, idx2, label in tqdm(all_pairs, desc="Evaluating pairs"):
        emb1 = embeddings_dict[idx1]
        emb2 = embeddings_dict[idx2]
        
        # Euclidean distance
        dist = F.pairwise_distance(emb1, emb2, p=2).item()
        distances.append(dist)
        labels_true.append(label)
    
    distances = np.array(distances)
    labels_true = np.array(labels_true)
    
    # Compute ROC curve
    # For ROC, we need to convert distance to similarity score
    # Lower distance = higher similarity, so we use 1 - normalized_distance
    max_dist = distances.max()
    min_dist = distances.min()
    if max_dist > min_dist:
        similarity_scores = 1 - (distances - min_dist) / (max_dist - min_dist)
    else:
        similarity_scores = np.ones_like(distances)
    
    fpr, tpr, roc_thresholds = roc_curve(labels_true, similarity_scores)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (maximize F1)
    best_f1 = 0
    best_threshold = threshold
    best_accuracy = 0
    best_precision = 0
    best_recall = 0
    for t in np.linspace(distances.min(), distances.max(), 100):
        pred = (distances < t).astype(int)
        f1_t = f1_score(labels_true, pred, zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t
            best_accuracy = accuracy_score(labels_true, pred)
            best_precision = precision_score(labels_true, pred, zero_division=0)
            best_recall = recall_score(labels_true, pred, zero_division=0)
    
    # Compute all metrics at optimal threshold
    labels_pred = (distances < best_threshold).astype(int)
    cm = confusion_matrix(labels_true, labels_pred)
    
    return {
        'accuracy': best_accuracy,
        'precision': best_precision,
        'recall': best_recall,
        'f1_score': best_f1,
        'roc_auc': roc_auc,
        'threshold': best_threshold,
        'confusion_matrix': cm.tolist(),
        'mean_distance_same': distances[labels_true == 1].mean(),
        'mean_distance_different': distances[labels_true == 0].mean(),
        'std_distance_same': distances[labels_true == 1].std(),
        'std_distance_different': distances[labels_true == 0].std(),
        'num_pairs': len(all_pairs),
        'num_positive': len(positive_pairs),
        'num_negative': len(negative_pairs),
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist(),
        'roc_thresholds': roc_thresholds.tolist()
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate models on test set')
    parser.add_argument('--model-path', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataset-path', type=str, default='data/lfw/lfw-filtered',
                        help='Path to LFW dataset')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--num-pairs', type=int, default=1000,
                        help='Number of pairs for verification evaluation')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Distance threshold for verification')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON file for results')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location='cpu')
    config_dict = checkpoint.get('config', {})
    
    model_name = config_dict.get('model_name', 'resnet18')
    embedding_size = config_dict.get('embedding_size', 128)
    
    if model_name == 'facenet':
        model = create_facenet_pretrained(
            embedding_size=embedding_size,
            pretrained=False
        ).to(device)
    else:
        model = create_face_net(
            embedding_size=embedding_size,
            pretrained=False
        ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    
    print(f"Loaded {model_name} model")
    print(f"  Embedding size: {embedding_size}")
    print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    print(f"  Best validation loss: {checkpoint.get('loss', 'unknown'):.4f}")
    
    # Load test dataset
    print(f"\nLoading test dataset from {args.dataset_path}...")
    test_transform = get_default_transforms(is_training=False)
    test_dataset = LFWDataset(
        dataset_path=args.dataset_path,
        transform=test_transform,
        split='test'
    )
    print(f"Test dataset: {len(test_dataset)} samples, {test_dataset.get_num_classes()} people")
    
    # Create dataloader for triplet loss evaluation
    test_sampler = BalancedTripletSampler(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        seed=42
    )
    test_loader = DataLoader(
        test_dataset,
        batch_sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate triplet loss
    print("\n" + "=" * 60)
    print("Evaluating Triplet Loss")
    print("=" * 60)
    loss_fn = TripletLoss(margin=0.2, distance_metric='euclidean')
    triplet_metrics = evaluate_triplet_loss(model, test_loader, loss_fn, device)
    
    print(f"\nTest Set Triplet Loss Metrics:")
    print(f"  Loss: {triplet_metrics['loss']:.4f}")
    print(f"  d(anchor, positive): {triplet_metrics['d_ap']:.4f}")
    print(f"  d(anchor, negative): {triplet_metrics['d_an']:.4f}")
    print(f"  Violation %: {triplet_metrics['violations']:.2%}")
    
    # Evaluate verification
    print("\n" + "=" * 60)
    print("Evaluating Face Verification")
    print("=" * 60)
    verification_metrics = evaluate_verification(
        model, test_dataset, device,
        num_pairs=args.num_pairs,
        threshold=args.threshold
    )
    
    print(f"\nTest Set Verification Metrics (at optimal threshold={verification_metrics['threshold']:.4f}):")
    print(f"  Accuracy: {verification_metrics['accuracy']:.4f}")
    print(f"  Precision: {verification_metrics['precision']:.4f}")
    print(f"  Recall: {verification_metrics['recall']:.4f}")
    print(f"  F1-Score: {verification_metrics['f1_score']:.4f}")
    print(f"  ROC-AUC: {verification_metrics['roc_auc']:.4f}")
    print(f"\nDistance Statistics:")
    print(f"  Mean distance (same person): {verification_metrics['mean_distance_same']:.4f} ± {verification_metrics['std_distance_same']:.4f}")
    print(f"  Mean distance (different person): {verification_metrics['mean_distance_different']:.4f} ± {verification_metrics['std_distance_different']:.4f}")
    
    print(f"\nConfusion Matrix:")
    cm = np.array(verification_metrics['confusion_matrix'])
    print(f"  True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
    print(f"  False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")
    
    # Combine results
    results = {
        'model_path': args.model_path,
        'model_name': model_name,
        'embedding_size': embedding_size,
        'triplet_metrics': triplet_metrics,
        'verification_metrics': verification_metrics
    }
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
    else:
        output_path = f"test_results_{model_name}.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

