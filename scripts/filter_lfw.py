#!/usr/bin/env python3
"""
Filter LFW dataset to only include people with 2+ images.
Creates a filtered copy of the dataset.
"""

import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def filter_lfw_dataset(
    source_path: str,
    dest_path: str,
    min_images: int = 2,
    copy_files: bool = True
):
    """
    Filter LFW dataset to only include people with minimum number of images.
    
    Args:
        source_path: Path to original LFW dataset
        dest_path: Path to save filtered dataset
        min_images: Minimum images per person (default: 2)
        copy_files: If True, copy files; if False, just create symlinks
    """
    source = Path(source_path)
    dest = Path(dest_path)
    
    if not source.exists():
        raise ValueError(f"Source path does not exist: {source_path}")
    
    # Create destination directory
    dest.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Filtering LFW Dataset")
    print("=" * 60)
    print(f"Source: {source_path}")
    print(f"Destination: {dest_path}")
    print(f"Minimum images per person: {min_images}")
    print()
    
    # Find all person folders
    person_folders = sorted([d for d in source.iterdir() if d.is_dir()])
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # Count images per person
    person_stats = {}
    total_people = 0
    total_images = 0
    filtered_people = 0
    filtered_images = 0
    
    print("Analyzing dataset...")
    for person_folder in person_folders:
        images = [
            f for f in person_folder.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        num_images = len(images)
        person_stats[person_folder] = {
            'name': person_folder.name,
            'images': images,
            'count': num_images
        }
        total_people += 1
        total_images += num_images
    
    print(f"Original dataset:")
    print(f"  Total people: {total_people}")
    print(f"  Total images: {total_images}")
    print()
    
    # Filter and copy
    print("Filtering and copying...")
    filtered_person_folders = []
    
    for person_folder, stats in person_stats.items():
        if stats['count'] >= min_images:
            # Create destination folder
            dest_person_folder = dest / stats['name']
            dest_person_folder.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            for img_path in stats['images']:
                dest_img_path = dest_person_folder / img_path.name
                
                if copy_files:
                    shutil.copy2(img_path, dest_img_path)
                else:
                    # Create symlink
                    try:
                        dest_img_path.symlink_to(img_path)
                    except OSError:
                        # If symlink fails, fall back to copy
                        shutil.copy2(img_path, dest_img_path)
                
                filtered_images += 1
            
            filtered_people += 1
            filtered_person_folders.append(stats['name'])
            
            if filtered_people % 100 == 0:
                print(f"  Processed {filtered_people} people...")
    
    print()
    print("=" * 60)
    print("Filtering Complete!")
    print("=" * 60)
    print(f"Filtered dataset:")
    print(f"  People included: {filtered_people} ({filtered_people/total_people*100:.1f}%)")
    print(f"  Images included: {filtered_images} ({filtered_images/total_images*100:.1f}%)")
    print(f"  People excluded: {total_people - filtered_people} ({(total_people-filtered_people)/total_people*100:.1f}%)")
    print(f"  Images excluded: {total_images - filtered_images} ({(total_images-filtered_images)/total_images*100:.1f}%)")
    print()
    print(f"Filtered dataset saved to: {dest_path}")
    print()
    print("Next steps:")
    print(f"  Update dataset path in your code to: {dest_path}")
    print(f"  Or use: python scripts/explore_lfw.py --dataset-path {dest_path} --visualize")
    
    # Save statistics
    stats_file = dest / "filtering_stats.txt"
    with open(stats_file, 'w') as f:
        f.write("LFW Dataset Filtering Statistics\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Minimum images per person: {min_images}\n\n")
        f.write(f"Original dataset:\n")
        f.write(f"  Total people: {total_people}\n")
        f.write(f"  Total images: {total_images}\n\n")
        f.write(f"Filtered dataset:\n")
        f.write(f"  People included: {filtered_people} ({filtered_people/total_people*100:.1f}%)\n")
        f.write(f"  Images included: {filtered_images} ({filtered_images/total_images*100:.1f}%)\n")
        f.write(f"  People excluded: {total_people - filtered_people} ({(total_people-filtered_people)/total_people*100:.1f}%)\n")
        f.write(f"  Images excluded: {total_images - filtered_images} ({(total_images-filtered_images)/total_images*100:.1f}%)\n")
    
    print(f"Statistics saved to: {stats_file}")
    
    return {
        'total_people': total_people,
        'total_images': total_images,
        'filtered_people': filtered_people,
        'filtered_images': filtered_images,
        'dest_path': str(dest_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Filter LFW dataset to only include people with 2+ images"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/lfw/lfw-dataset",
        help="Path to original LFW dataset (default: data/lfw/lfw-dataset)"
    )
    parser.add_argument(
        "--dest",
        type=str,
        default="data/lfw/lfw-filtered",
        help="Path to save filtered dataset (default: data/lfw/lfw-filtered)"
    )
    parser.add_argument(
        "--min-images",
        type=int,
        default=2,
        help="Minimum images per person (default: 2)"
    )
    parser.add_argument(
        "--symlink",
        action="store_true",
        help="Create symlinks instead of copying files (saves disk space)"
    )
    
    args = parser.parse_args()
    
    filter_lfw_dataset(
        source_path=args.source,
        dest_path=args.dest,
        min_images=args.min_images,
        copy_files=not args.symlink
    )


if __name__ == "__main__":
    main()

