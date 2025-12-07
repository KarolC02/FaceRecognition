#!/usr/bin/env python3
"""
Download and extract LFW (Labeled Faces in the Wild) dataset.
Uses kagglehub for downloading.
"""

import os
import shutil
from pathlib import Path
import argparse

try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False
    print("Warning: kagglehub not available. Install with: pip install kagglehub")


def download_lfw_dataset(data_dir="data"):
    """
    Download LFW dataset using kagglehub.
    
    Args:
        data_dir: Directory to store the dataset
    """
    if not KAGGLEHUB_AVAILABLE:
        raise ImportError(
            "kagglehub is required. Install with: pip install kagglehub"
        )
    
    data_path = Path(data_dir)
    lfw_path = data_path / "lfw"
    lfw_path.mkdir(parents=True, exist_ok=True)
    
    print("Downloading LFW dataset from Kaggle...")
    print("This may take a few minutes...")
    
    try:
        # Download latest version using kagglehub
        path = kagglehub.dataset_download("jessicali9530/lfw-dataset")
        print(f"✓ Dataset downloaded to: {path}")
        
        # The dataset might be in a subdirectory, let's find the actual data
        dataset_path = Path(path)
        
        # Look for lfw-deepfunneled or lfw directory
        possible_paths = [
            dataset_path / "lfw-deepfunneled",
            dataset_path / "lfw",
            dataset_path,
        ]
        
        # Also check subdirectories
        for subdir in dataset_path.iterdir():
            if subdir.is_dir():
                possible_paths.extend([
                    subdir / "lfw-deepfunneled",
                    subdir / "lfw",
                    subdir,
                ])
        
        # Find the directory with person-named subdirectories
        actual_dataset_path = None
        for possible_path in possible_paths:
            if possible_path.exists() and possible_path.is_dir():
                # Check if it has person-named subdirectories (typical LFW structure)
                subdirs = [d for d in possible_path.iterdir() if d.is_dir()]
                if subdirs and len(subdirs) > 10:  # LFW has many people
                    actual_dataset_path = possible_path
                    break
        
        if actual_dataset_path is None:
            # If we can't find the structure, use the root
            actual_dataset_path = dataset_path
            print(f"⚠ Could not determine exact dataset structure, using: {actual_dataset_path}")
        
        # Create symlink or copy to our data directory
        target_path = lfw_path / "lfw-dataset"
        if target_path.exists():
            print(f"Dataset already linked at: {target_path}")
        else:
            # Create a symlink (or copy if symlink fails)
            try:
                target_path.symlink_to(actual_dataset_path)
                print(f"✓ Created symlink: {target_path} -> {actual_dataset_path}")
            except OSError:
                # If symlink fails (e.g., on Windows or cross-filesystem), copy
                print(f"Copying dataset to {target_path}...")
                shutil.copytree(actual_dataset_path, target_path)
                print(f"✓ Copied dataset to: {target_path}")
        
        print(f"\n✓ LFW dataset ready!")
        print(f"  Location: {target_path}")
        return str(target_path)
        
    except Exception as e:
        print(f"\n✗ Error downloading dataset: {e}")
        print("\nMake sure you have kagglehub installed and configured:")
        print("  pip install kagglehub")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download LFW dataset using kagglehub")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Directory to store the dataset (default: data)"
    )
    
    args = parser.parse_args()
    
    dataset_path = download_lfw_dataset(args.data_dir)
    print(f"\n✓ Dataset downloaded successfully!")
    print(f"  Location: {dataset_path}")
    print(f"\nNext step: Run explore_lfw.py to verify the dataset format:")
    print(f"  python scripts/explore_lfw.py --dataset-path {dataset_path} --visualize")


if __name__ == "__main__":
    main()

