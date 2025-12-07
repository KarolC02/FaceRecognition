"""
Face Recognition Network based on ResNet18.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from typing import Optional


class FaceNet(nn.Module):
    """
    Face recognition network using ResNet18 as backbone.
    
    The network outputs a fixed-size embedding vector (e.g., 128 or 512 dimensions)
    that can be used for face recognition via distance metrics.
    """
    
    def __init__(
        self,
        embedding_size: int = 128,
        pretrained: bool = True,
        num_classes: Optional[int] = None,
    ):
        """
        Initialize FaceNet.
        
        Args:
            embedding_size: Size of the face embedding vector (default: 128)
            pretrained: Whether to use ImageNet pretrained weights (default: True)
            num_classes: If provided, adds a classification head (optional)
        """
        super(FaceNet, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        
        # Load ResNet18 backbone
        # Use new weights API instead of deprecated pretrained parameter
        if pretrained:
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet = models.resnet18(weights=None)
        
        # Remove the final fully connected layer
        # ResNet18's features: conv layers + avgpool + fc
        # We'll keep everything except the fc layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # Remove fc layer
        
        # Get the feature size from ResNet18 (512 for ResNet18)
        feature_size = 512
        
        # Add embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(feature_size, embedding_size),
            nn.BatchNorm1d(embedding_size),
        )
        
        # Optional classification head (for training with classification loss)
        if num_classes is not None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = None
    
    def forward(self, x: torch.Tensor, return_embedding: bool = True) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input images [batch_size, 3, height, width]
            return_embedding: If True, return embedding; else return classification logits
        
        Returns:
            If return_embedding=True: embedding [batch_size, embedding_size]
            If return_embedding=False and classifier exists: logits [batch_size, num_classes]
        """
        # Extract features using ResNet18 backbone
        features = self.backbone(x)  # [batch_size, 512, 1, 1]
        features = features.view(features.size(0), -1)  # [batch_size, 512]
        
        # Generate embedding
        embedding = self.embedding(features)  # [batch_size, embedding_size]
        
        # Normalize embedding (L2 normalization for cosine similarity)
        embedding = nn.functional.normalize(embedding, p=2, dim=1)
        
        if return_embedding:
            return embedding
        elif self.classifier is not None:
            logits = self.classifier(embedding)
            return logits
        else:
            return embedding
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get face embedding for input images.
        
        Args:
            x: Input images [batch_size, 3, height, width]
        
        Returns:
            embedding: [batch_size, embedding_size]
        """
        self.eval()
        with torch.no_grad():
            embedding = self.forward(x, return_embedding=True)
        return embedding


def create_face_net(
    embedding_size: int = 128,
    pretrained: bool = True,
    num_classes: Optional[int] = None,
) -> FaceNet:
    """
    Factory function to create a FaceNet model.
    
    Args:
        embedding_size: Size of embedding vector
        pretrained: Use ImageNet pretrained weights
        num_classes: Optional number of classes for classification head
    
    Returns:
        FaceNet model
    """
    model = FaceNet(
        embedding_size=embedding_size,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


if __name__ == "__main__":
    # Test the model
    print("Testing FaceNet model...")
    
    # Create model
    model = create_face_net(embedding_size=128, pretrained=True)
    print(f"Model created with embedding size: {model.embedding_size}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    # Get embedding
    embedding = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm (should be ~1.0): {embedding.norm(dim=1).mean().item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

