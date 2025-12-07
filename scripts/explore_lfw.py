#!/usr/bin/env python3
"""
Explore and analyze LFW dataset structure and format.
"""

import os
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


def analyze_lfw_structure(dataset_path):
    """Analyze the structure of LFW dataset."""
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return
    
    print("=" * 60)
    print("LFW Dataset Structure Analysis")
    print("=" * 60)
    
    # Check if it's organized by person folders
    person_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if person_folders:
        print(f"\n✓ Dataset is organized by person folders")
        print(f"  Number of people: {len(person_folders)}")
        
        # Count images per person
        images_per_person = []
        total_images = 0
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for person_folder in person_folders:
            images = [f for f in person_folder.iterdir() 
                     if f.suffix.lower() in image_extensions]
            num_images = len(images)
            images_per_person.append(num_images)
            total_images += num_images
        
        print(f"  Total images: {total_images}")
        print(f"  Average images per person: {np.mean(images_per_person):.2f}")
        print(f"  Min images per person: {min(images_per_person)}")
        print(f"  Max images per person: {max(images_per_person)}")
        print(f"  Median images per person: {np.median(images_per_person):.2f}")
        
        # Distribution of images per person
        distribution = Counter(images_per_person)
        print(f"\n  Distribution of images per person:")
        for num_img in sorted(distribution.keys())[:10]:
            count = distribution[num_img]
            print(f"    {num_img} images: {count} people")
        if len(distribution) > 10:
            print(f"    ... (showing top 10)")
        
        # Show some example person folders
        print(f"\n  Example person folders (first 5):")
        for person_folder in sorted(person_folders)[:5]:
            num_images = len([f for f in person_folder.iterdir() 
                            if f.suffix.lower() in image_extensions])
            print(f"    {person_folder.name}: {num_images} images")
        
        # Check image properties
        print(f"\n  Analyzing sample images...")
        sample_images = []
        for person_folder in person_folders[:10]:  # Check first 10 people
            images = [f for f in person_folder.iterdir() 
                     if f.suffix.lower() in image_extensions]
            if images:
                sample_images.append(images[0])
        
        if sample_images:
            sizes = []
            modes = []
            for img_path in sample_images[:20]:  # Check 20 sample images
                try:
                    img = Image.open(img_path)
                    sizes.append(img.size)
                    modes.append(img.mode)
                except Exception as e:
                    print(f"    Warning: Could not open {img_path}: {e}")
            
            if sizes:
                print(f"    Image sizes (sample): {set(sizes)}")
                print(f"    Image modes: {set(modes)}")
                avg_width = np.mean([s[0] for s in sizes])
                avg_height = np.mean([s[1] for s in sizes])
                print(f"    Average size: {avg_width:.0f}x{avg_height:.0f}")
        
        return {
            'num_people': len(person_folders),
            'total_images': total_images,
            'images_per_person': images_per_person,
            'structure': 'person_folders'
        }
    
    else:
        # Check if it's a flat structure
        image_files = [f for f in dataset_path.iterdir() 
                      if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}]
        
        if image_files:
            print(f"\n✓ Dataset is a flat structure")
            print(f"  Total images: {len(image_files)}")
            print(f"  Note: Flat structure - may need to parse filenames for person labels")
            return {
                'total_images': len(image_files),
                'structure': 'flat'
            }
        else:
            print(f"\n✗ Could not determine dataset structure")
            print(f"  Path exists but no person folders or images found")
            return None


def visualize_sample_images(dataset_path, num_samples=9):
    """Visualize sample images from the dataset."""
    dataset_path = Path(dataset_path)
    person_folders = [d for d in dataset_path.iterdir() if d.is_dir()]
    
    if not person_folders:
        print("Cannot visualize: No person folders found")
        return
    
    # Collect sample images
    sample_images = []
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    for person_folder in person_folders[:num_samples]:
        images = [f for f in person_folder.iterdir() 
                 if f.suffix.lower() in image_extensions]
        if images:
            sample_images.append((person_folder.name, images[0]))
    
    if not sample_images:
        print("No images found to visualize")
        return
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    for idx, (person_name, img_path) in enumerate(sample_images[:num_samples]):
        try:
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(f"{person_name}\n{img_path.name}", fontsize=8)
            axes[idx].axis('off')
        except Exception as e:
            axes[idx].text(0.5, 0.5, f"Error loading\n{img_path.name}", 
                          ha='center', va='center')
            axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = Path("data") / "lfw_samples.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Sample images saved to: {output_path}")
    plt.close()


def check_pairs_file(pairs_path):
    """Check if pairs.txt file exists and analyze it."""
    pairs_path = Path(pairs_path)
    
    if not pairs_path.exists():
        print(f"\n⚠ pairs.txt not found at {pairs_path}")
        print("  This file is used for LFW evaluation protocol")
        return
    
    print(f"\n✓ Found pairs.txt")
    with open(pairs_path, 'r') as f:
        lines = f.readlines()
    
    print(f"  Total lines: {len(lines)}")
    print(f"  First few lines:")
    for line in lines[:5]:
        print(f"    {line.strip()}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore LFW dataset")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/lfw/lfw-deepfunneled",
        help="Path to LFW dataset (default: data/lfw/lfw-deepfunneled)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualization of sample images"
    )
    
    args = parser.parse_args()
    
    # Analyze structure
    stats = analyze_lfw_structure(args.dataset_path)
    
    # Check pairs file
    pairs_path = Path(args.dataset_path).parent / "pairs.txt"
    check_pairs_file(pairs_path)
    
    # Visualize if requested
    if args.visualize:
        visualize_sample_images(args.dataset_path)
    
    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

