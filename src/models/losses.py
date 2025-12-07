"""
Loss functions for face recognition training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class TripletLoss(nn.Module):
    """
    Triplet Loss for metric learning.
    
    For each anchor, we want:
    d(anchor, positive) < d(anchor, negative) + margin
    
    Where d is the distance (typically Euclidean or cosine).
    """
    
    def __init__(self, margin: float = 0.2, distance_metric: str = "euclidean"):
        """
        Initialize Triplet Loss.
        
        Args:
            margin: Margin for triplet loss (default: 0.2)
            distance_metric: 'euclidean' or 'cosine' (default: 'euclidean')
        """
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
        
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with loss components for logging
        """
        if self.distance_metric == "euclidean":
            # Euclidean distance (L2)
            d_ap = F.pairwise_distance(anchor, positive, p=2)  # [batch_size]
            d_an = F.pairwise_distance(anchor, negative, p=2)  # [batch_size]
        elif self.distance_metric == "cosine":
            # Cosine distance (1 - cosine_similarity)
            # Since embeddings are already normalized, cosine distance = 1 - dot product
            d_ap = 1 - F.cosine_similarity(anchor, positive)  # [batch_size]
            d_an = 1 - F.cosine_similarity(anchor, negative)  # [batch_size]
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Triplet loss: max(0, d(anchor, positive) - d(anchor, negative) + margin)
        loss = F.relu(d_ap - d_an + self.margin)
        
        # Average over batch
        loss = loss.mean()
        
        # Compute metrics for logging
        metrics = {
            'loss': loss.item(),
            'd_ap_mean': d_ap.mean().item(),
            'd_an_mean': d_an.mean().item(),
            'margin_violations': (d_ap > d_an - self.margin).float().mean().item(),
            'hard_triplets': (d_ap > d_an - self.margin).sum().item(),
        }
        
        return loss, metrics


class HardTripletLoss(nn.Module):
    """
    Hard Triplet Loss - focuses on hard negatives.
    
    Instead of using all triplets, this uses:
    - Hard negatives: negatives that are close to anchor
    - Semi-hard negatives: negatives that are further than positive but still violate margin
    """
    
    def __init__(self, margin: float = 0.2, distance_metric: str = "euclidean", 
                 mining_strategy: str = "hard"):
        """
        Initialize Hard Triplet Loss.
        
        Args:
            margin: Margin for triplet loss
            distance_metric: 'euclidean' or 'cosine'
            mining_strategy: 'hard', 'semi-hard', or 'all'
                - 'hard': Only use hard negatives (d_an < d_ap)
                - 'semi-hard': Use semi-hard negatives (d_ap < d_an < d_ap + margin)
                - 'all': Use all triplets
        """
        super(HardTripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
        self.mining_strategy = mining_strategy
        self.base_loss = TripletLoss(margin, distance_metric)
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute hard triplet loss.
        
        Args:
            anchor: Anchor embeddings [batch_size, embedding_dim]
            positive: Positive embeddings [batch_size, embedding_dim]
            negative: Negative embeddings [batch_size, embedding_dim]
        
        Returns:
            loss: Scalar loss value
            metrics: Dictionary with loss components
        """
        if self.distance_metric == "euclidean":
            d_ap = F.pairwise_distance(anchor, positive, p=2)
            d_an = F.pairwise_distance(anchor, negative, p=2)
        elif self.distance_metric == "cosine":
            d_ap = 1 - F.cosine_similarity(anchor, positive)
            d_an = 1 - F.cosine_similarity(anchor, negative)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Apply mining strategy
        if self.mining_strategy == "hard":
            # Hard negatives: d_an < d_ap (negative is closer than positive)
            mask = d_an < d_ap
        elif self.mining_strategy == "semi-hard":
            # Semi-hard: d_ap < d_an < d_ap + margin
            mask = (d_ap < d_an) & (d_an < d_ap + self.margin)
        else:  # "all"
            mask = torch.ones_like(d_ap, dtype=torch.bool)
        
        if mask.sum() == 0:
            # No hard triplets found, return zero loss
            return torch.tensor(0.0, device=anchor.device, requires_grad=True), {
                'loss': 0.0,
                'hard_triplets': 0,
                'total_triplets': len(d_ap),
                'hard_ratio': 0.0
            }
        
        # Filter to hard triplets only
        anchor_hard = anchor[mask]
        positive_hard = positive[mask]
        negative_hard = negative[mask]
        
        # Compute loss on hard triplets
        loss, base_metrics = self.base_loss(anchor_hard, positive_hard, negative_hard)
        
        metrics = {
            **base_metrics,
            'hard_triplets': mask.sum().item(),
            'total_triplets': len(d_ap),
            'hard_ratio': mask.float().mean().item()
        }
        
        return loss, metrics


if __name__ == "__main__":
    # Test triplet loss
    print("Testing Triplet Loss...")
    
    batch_size = 8
    embedding_dim = 128
    
    # Create dummy embeddings
    anchor = torch.randn(batch_size, embedding_dim)
    positive = anchor + 0.1 * torch.randn(batch_size, embedding_dim)  # Close to anchor
    negative = torch.randn(batch_size, embedding_dim)  # Far from anchor
    
    # Normalize embeddings (as our model does)
    anchor = F.normalize(anchor, p=2, dim=1)
    positive = F.normalize(positive, p=2, dim=1)
    negative = F.normalize(negative, p=2, dim=1)
    
    # Test standard triplet loss
    triplet_loss = TripletLoss(margin=0.2, distance_metric="euclidean")
    loss, metrics = triplet_loss(anchor, positive, negative)
    
    print(f"\nStandard Triplet Loss:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  d(anchor, positive): {metrics['d_ap_mean']:.4f}")
    print(f"  d(anchor, negative): {metrics['d_an_mean']:.4f}")
    print(f"  Margin violations: {metrics['margin_violations']:.2%}")
    
    # Test hard triplet loss
    hard_loss = HardTripletLoss(margin=0.2, mining_strategy="hard")
    loss_hard, metrics_hard = hard_loss(anchor, positive, negative)
    
    print(f"\nHard Triplet Loss:")
    print(f"  Loss: {loss_hard.item():.4f}")
    print(f"  Hard triplets: {metrics_hard['hard_triplets']}/{metrics_hard['total_triplets']}")
    print(f"  Hard ratio: {metrics_hard['hard_ratio']:.2%}")
    
    print("\n✓ Triplet loss tests passed!")

