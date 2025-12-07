"""
PyTorch Dataset class for LFW (Labeled Faces in the Wild).
"""

import os
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class LFWDataset(Dataset):
    """
    LFW Dataset for face recognition training.
    
    The dataset is organized as:
        dataset_path/
            person1_name/
                image1.jpg
                image2.jpg
            person2_name/
                ...
    """
    
    def __init__(
        self,
        dataset_path: str = "data/lfw/lfw-filtered",  # Default to filtered dataset
        transform: Optional[transforms.Compose] = None,
        min_images_per_person: int = 1,  # Not used if dataset is already filtered
        split: str = "train",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ):
        """
        Initialize LFW Dataset.
        
        Args:
            dataset_path: Path to LFW dataset root (contains person-named folders)
            transform: Optional transform to apply to images
            min_images_per_person: Minimum images per person to include (default: 1)
            split: 'train', 'val', or 'test'
            train_ratio: Ratio of data for training (default: 0.8)
            val_ratio: Ratio of data for validation (default: 0.1)
            seed: Random seed for reproducible splits
        """
        self.dataset_path = Path(dataset_path)
        self.transform = transform
        self.min_images_per_person = min_images_per_person
        self.split = split
        
        # Load all image paths with labels
        self.samples = self._load_samples()
        
        # Split dataset
        self.samples = self._split_dataset(self.samples, train_ratio, val_ratio, seed)
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"  Unique people: {len(set(label for _, label in self.samples))}")
    
    def _load_samples(self) -> List[Tuple[str, int]]:
        """Load all image paths and their person labels."""
        samples = []
        person_to_id = {}
        person_id = 0
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        # Iterate through person folders
        person_folders = sorted([d for d in self.dataset_path.iterdir() if d.is_dir()])
        
        for person_folder in person_folders:
            # Get all images for this person
            images = [
                f for f in person_folder.iterdir()
                if f.suffix.lower() in image_extensions
            ]
            
            # Filter by minimum images per person
            if len(images) < self.min_images_per_person:
                continue
            
            # Assign person ID
            person_name = person_folder.name
            if person_name not in person_to_id:
                person_to_id[person_name] = person_id
                person_id += 1
            
            person_label = person_to_id[person_name]
            
            # Add all images for this person
            for img_path in images:
                samples.append((str(img_path), person_label))
        
        self.person_to_id = person_to_id
        self.id_to_person = {v: k for k, v in person_to_id.items()}
        
        return samples
    
    def _split_dataset(
        self,
        samples: List[Tuple[str, int]],
        train_ratio: float,
        val_ratio: float,
        seed: int
    ) -> List[Tuple[str, int]]:
        """Split dataset by person (not by image) to avoid data leakage."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Group samples by person
        person_samples: Dict[int, List[Tuple[str, int]]] = {}
        for img_path, label in samples:
            if label not in person_samples:
                person_samples[label] = []
            person_samples[label].append((img_path, label))
        
        # Get list of unique people
        people = list(person_samples.keys())
        np.random.shuffle(people)
        
        # Calculate split indices
        n_people = len(people)
        n_train = int(n_people * train_ratio)
        n_val = int(n_people * val_ratio)
        
        # Split people
        if self.split == "train":
            selected_people = people[:n_train]
        elif self.split == "val":
            selected_people = people[n_train:n_train + n_val]
        else:  # test
            selected_people = people[n_train + n_val:]
        
        # Get all samples for selected people
        split_samples = []
        for person_id in selected_people:
            split_samples.extend(person_samples[person_id])
        
        return split_samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Returns:
            image: PIL Image or transformed tensor
            label: Person ID (integer)
        """
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (250, 250), color='black')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_num_classes(self) -> int:
        """Get number of unique people in the dataset."""
        return len(set(label for _, label in self.samples))


def get_default_transforms(image_size: int = 224, is_training: bool = True):
    """
    Get default transforms for LFW dataset.
    
    Args:
        image_size: Target image size (default: 224 for ResNet18)
        is_training: If True, apply data augmentation
    
    Returns:
        transforms.Compose: Image transforms
    """
    if is_training:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


if __name__ == "__main__":
    # Test the dataset
    dataset_path = "data/lfw/lfw-dataset"
    
    print("Testing LFW Dataset...")
    
    # Create train dataset
    train_transform = get_default_transforms(is_training=True)
    train_dataset = LFWDataset(
        dataset_path=dataset_path,
        transform=train_transform,
        split="train"
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    
    # Test getting a sample
    if len(train_dataset) > 0:
        image, label = train_dataset[0]
        print(f"Sample image shape: {image.shape}")
        print(f"Sample label: {label}")
        print(f"Person name: {train_dataset.id_to_person[label]}")

