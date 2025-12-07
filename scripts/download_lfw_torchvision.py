#!/usr/bin/env python3
"""
Alternative: Download LFW using PyTorch's torchvision (if available).
Note: torchvision's LFW is for pairs evaluation, not full dataset.
This script checks if we can use it as an alternative.
"""

import os
from pathlib import Path

try:
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    TORCHVISION_AVAILABLE = True
except ImportError:
    TORCHVISION_AVAILABLE = False
    print("torchvision not available")


def check_torchvision_lfw(data_dir="data"):
    """Check if torchvision can download LFW pairs dataset."""
    if not TORCHVISION_AVAILABLE:
        print("torchvision is not installed")
        return False
    
    data_path = Path(data_dir)
    lfw_path = data_path / "lfw"
    lfw_path.mkdir(parents=True, exist_ok=True)
    
    print("Note: torchvision.datasets.LFWPairs downloads pairs for evaluation,")
    print("      not the full dataset for training.")
    print("      For training, you need the full dataset from the official website.")
    
    return False


if __name__ == "__main__":
    check_torchvision_lfw()

