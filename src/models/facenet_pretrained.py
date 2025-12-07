"""
FaceNet model using pretrained Inception-ResNet-v1 architecture.
This is the architecture used in the original FaceNet paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BasicConv2d(nn.Module):
    """Basic convolutional block with batch normalization."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class Block35(nn.Module):
    """Inception-ResNet block."""
    
    def __init__(self, scale=1.0):
        super(Block35, self).__init__()
        self.scale = scale
        
        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )
        
        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):
    """Inception-ResNet block for 17x17 feature maps."""
    
    def __init__(self, scale=1.0):
        super(Block17, self).__init__()
        self.scale = scale
        
        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0))
        )
        
        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):
    """Inception-ResNet block for 8x8 feature maps."""
    
    def __init__(self, scale=1.0, noReLU=False):
        super(Block8, self).__init__()
        self.scale = scale
        self.noReLU = noReLU
        
        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0))
        )
        
        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):
    """Mixed block for transition."""
    
    def __init__(self):
        super(Mixed_6a, self).__init__()
        
        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )
        
        self.branch2 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):
    """Mixed block for transition."""
    
    def __init__(self):
        super(Mixed_7a, self).__init__()
        
        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )
        
        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )
        
        self.branch3 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(nn.Module):
    """Inception-ResNet-v1 architecture (FaceNet backbone)."""
    
    def __init__(self, classify=False, num_classes=None, dropout_prob=0.6):
        super(InceptionResnetV1, self).__init__()
        
        # Stem
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        
        # Inception-ResNet blocks
        self.repeat_1 = nn.Sequential(*[Block35(scale=0.17) for _ in range(5)])
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(*[Block17(scale=0.10) for _ in range(10)])
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(*[Block8(scale=0.20) for _ in range(5)])
        self.block8 = Block8(noReLU=True)
        
        # Final layers
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        
        self.classify = classify
        if classify:
            self.logits = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Stem
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        
        # Inception-ResNet blocks
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        
        # Final layers
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.last_linear(x)
        x = self.last_bn(x)
        
        # L2 normalize
        x = F.normalize(x, p=2, dim=1)
        
        if self.classify:
            x = self.logits(x)
        
        return x


class FaceNetPretrained(nn.Module):
    """
    FaceNet model using Inception-ResNet-v1 architecture.
    Can load pretrained weights or train from scratch.
    """
    
    def __init__(
        self,
        embedding_size: int = 512,
        pretrained: bool = False,
        pretrained_path: Optional[str] = None,
        num_classes: Optional[int] = None,
    ):
        """
        Initialize FaceNet.
        
        Args:
            embedding_size: Size of face embedding (512 for FaceNet)
            pretrained: Whether to use pretrained weights
            pretrained_path: Path to pretrained weights file
            num_classes: If provided, adds classification head
        """
        super(FaceNetPretrained, self).__init__()
        
        self.embedding_size = embedding_size
        self.num_classes = num_classes
        
        # Create Inception-ResNet-v1 backbone
        self.backbone = InceptionResnetV1(classify=False, num_classes=None)
        
        # If embedding size is not 512, add a projection layer
        if embedding_size != 512:
            self.projection = nn.Sequential(
                nn.Linear(512, embedding_size),
                nn.BatchNorm1d(embedding_size),
            )
        else:
            self.projection = None
        
        # Classification head (optional)
        if self.num_classes is not None:
            self.classifier = nn.Linear(embedding_size, num_classes)
        else:
            self.classifier = None
        
        # Load pretrained weights if specified
        if pretrained and pretrained_path:
            self.load_pretrained(pretrained_path)
        elif pretrained:
            print("Warning: pretrained=True but no pretrained_path provided")
            print("  Training from scratch...")
    
    def load_pretrained(self, pretrained_path: str):
        """Load pretrained weights."""
        try:
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
            else:
                state_dict = checkpoint
            
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load backbone weights (filter to match our architecture)
            backbone_dict = {}
            for k, v in state_dict.items():
                if k.startswith('backbone.') or not k.startswith('projection') and not k.startswith('classifier'):
                    key = k.replace('backbone.', '')
                    if key in self.backbone.state_dict():
                        backbone_dict[key] = v
            
            self.backbone.load_state_dict(backbone_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("  Training from scratch...")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch_size, 3, 224, 224)
        
        Returns:
            Embedding vector (batch_size, embedding_size)
        """
        # Get features from backbone
        x = self.backbone(x)
        
        # Project to desired embedding size if needed
        if self.projection is not None:
            x = self.projection(x)
            x = F.normalize(x, p=2, dim=1)
        
        # Classification head (optional)
        if self.classifier is not None:
            x = self.classifier(x)
        
        return x


def create_facenet_pretrained(
    embedding_size: int = 512,
    pretrained: bool = False,
    pretrained_path: Optional[str] = None,
    num_classes: Optional[int] = None,
) -> FaceNetPretrained:
    """
    Create a FaceNet model.
    
    Args:
        embedding_size: Size of face embedding (default: 512)
        pretrained: Whether to use pretrained weights
        pretrained_path: Path to pretrained weights
        num_classes: If provided, adds classification head
    
    Returns:
        FaceNet model
    """
    model = FaceNetPretrained(
        embedding_size=embedding_size,
        pretrained=pretrained,
        pretrained_path=pretrained_path,
        num_classes=num_classes,
    )
    return model

