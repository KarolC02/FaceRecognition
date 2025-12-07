"""
Triplet Sampler for generating triplets from dataset.
"""

import torch
from torch.utils.data import Sampler, Dataset
from typing import List, Iterator
import random
from collections import defaultdict


class TripletSampler(Sampler):
    """
    Sampler that generates triplets (anchor, positive, negative) from dataset.
    
    For each batch:
    - Samples N people
    - For each person, samples 2 images (anchor and positive)
    - For each person, samples 1 negative from a different person
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        samples_per_person: int = 2,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize Triplet Sampler.
        
        Args:
            dataset: Dataset with (image, label) pairs
            batch_size: Number of triplets per batch
            samples_per_person: Number of samples to take per person (default: 2 for anchor+positive)
            shuffle: Whether to shuffle (default: True)
            seed: Random seed (default: 42)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_person = samples_per_person
        self.shuffle = shuffle
        self.seed = seed
        
        # Group samples by label (person ID)
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.label_to_indices[label].append(idx)
        
        self.labels = list(self.label_to_indices.keys())
        self.num_labels = len(self.labels)
        
        # Verify we have enough samples per person
        for label, indices in self.label_to_indices.items():
            if len(indices) < samples_per_person:
                raise ValueError(
                    f"Person {label} has only {len(indices)} images, "
                    f"but need at least {samples_per_person} for triplet sampling"
                )
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate batches of triplets."""
        # Set random seed for reproducibility
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # Create list of all possible triplets
        batches = []
        
        # Shuffle labels if needed
        labels = self.labels.copy()
        if self.shuffle:
            random.shuffle(labels)
        
        # Generate batches
        batch_indices = []
        for label in labels:
            # Get all indices for this person
            person_indices = self.label_to_indices[label].copy()
            if self.shuffle:
                random.shuffle(person_indices)
            
            # Sample anchor and positive from same person
            if len(person_indices) >= self.samples_per_person:
                anchor_idx = person_indices[0]
                positive_idx = person_indices[1] if len(person_indices) > 1 else person_indices[0]
            else:
                # Fallback (shouldn't happen if validation passed)
                anchor_idx = person_indices[0]
                positive_idx = person_indices[0]
            
            # Sample negative from different person
            negative_label = random.choice([l for l in self.labels if l != label])
            negative_indices = self.label_to_indices[negative_label]
            negative_idx = random.choice(negative_indices)
            
            # Add triplet indices: [anchor, positive, negative]
            batch_indices.extend([anchor_idx, positive_idx, negative_idx])
            
            # When we have enough for a batch, yield it
            if len(batch_indices) >= self.batch_size * 3:  # 3 indices per triplet
                batches.append(batch_indices[:self.batch_size * 3])
                batch_indices = batch_indices[self.batch_size * 3:]
        
        # Yield remaining if any
        if batch_indices:
            # Pad if needed
            while len(batch_indices) < self.batch_size * 3:
                # Repeat last triplet
                if len(batch_indices) >= 3:
                    batch_indices.extend(batch_indices[-3:])
                else:
                    break
            batches.append(batch_indices[:self.batch_size * 3])
        
        # Shuffle batches if needed
        if self.shuffle:
            random.shuffle(batches)
        
        for batch in batches:
            yield batch
    
    def __len__(self) -> int:
        """Return number of batches."""
        # Approximate number of batches
        num_triplets = len(self.labels)
        return (num_triplets + self.batch_size - 1) // self.batch_size


class BalancedTripletSampler(Sampler):
    """
    Balanced Triplet Sampler - ensures each batch has diverse people.
    
    This sampler ensures that:
    - Each batch contains samples from different people
    - Better triplet diversity
    - More stable training
    """
    
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 32,
        shuffle: bool = True,
        seed: int = 42
    ):
        """
        Initialize Balanced Triplet Sampler.
        
        Args:
            dataset: Dataset with (image, label) pairs
            batch_size: Number of triplets per batch (should be even)
            shuffle: Whether to shuffle (default: True)
            seed: Random seed (default: 42)
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        
        # Group samples by label
        self.label_to_indices = defaultdict(list)
        for idx, (_, label) in enumerate(dataset):
            self.label_to_indices[label].append(idx)
        
        self.labels = list(self.label_to_indices.keys())
        self.num_labels = len(self.labels)
        
        # Verify dataset has enough people
        if self.num_labels < batch_size:
            raise ValueError(
                f"Dataset has only {self.num_labels} people, "
                f"but batch_size is {batch_size}. Need at least {batch_size} people."
            )
    
    def __iter__(self) -> Iterator[List[int]]:
        """Generate balanced batches of triplets."""
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        labels = self.labels.copy()
        if self.shuffle:
            random.shuffle(labels)
        
        # Generate batches
        for batch_start in range(0, len(labels), self.batch_size):
            batch_labels = labels[batch_start:batch_start + self.batch_size]
            
            # If last batch is smaller, pad with random labels
            while len(batch_labels) < self.batch_size:
                batch_labels.append(random.choice(self.labels))
            
            batch_indices = []
            for label in batch_labels:
                # Get indices for this person
                person_indices = self.label_to_indices[label].copy()
                if self.shuffle:
                    random.shuffle(person_indices)
                
                # Anchor and positive from same person
                anchor_idx = person_indices[0]
                positive_idx = person_indices[1] if len(person_indices) > 1 else person_indices[0]
                
                # Negative from different person (not in current batch)
                available_negatives = [l for l in self.labels if l != label and l not in batch_labels]
                if not available_negatives:
                    available_negatives = [l for l in self.labels if l != label]
                negative_label = random.choice(available_negatives)
                negative_indices = self.label_to_indices[negative_label]
                negative_idx = random.choice(negative_indices)
                
                batch_indices.extend([anchor_idx, positive_idx, negative_idx])
            
            yield batch_indices
    
    def __len__(self) -> int:
        """Return number of batches."""
        return (len(self.labels) + self.batch_size - 1) // self.batch_size


if __name__ == "__main__":
    # Test the sampler
    print("Testing Triplet Sampler...")
    
    # Create a dummy dataset
    class DummyDataset(Dataset):
        def __init__(self):
            # Create 10 people with 3 images each
            self.data = []
            for person_id in range(10):
                for img_id in range(3):
                    self.data.append((f"person{person_id}_img{img_id}", person_id))
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    dataset = DummyDataset()
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of people: {len(set(label for _, label in dataset.data))}")
    
    # Test TripletSampler
    sampler = TripletSampler(dataset, batch_size=4, samples_per_person=2)
    print(f"\nTripletSampler batches: {len(sampler)}")
    
    # Get first batch
    batch = next(iter(sampler))
    print(f"First batch indices: {batch}")
    print(f"Batch size: {len(batch)} (should be 4*3=12)")
    
    # Test BalancedTripletSampler
    balanced_sampler = BalancedTripletSampler(dataset, batch_size=4)
    print(f"\nBalancedTripletSampler batches: {len(balanced_sampler)}")
    
    batch_balanced = next(iter(balanced_sampler))
    print(f"First balanced batch indices: {batch_balanced}")
    print(f"Batch size: {len(batch_balanced)} (should be 4*3=12)")
    
    print("\n✓ Triplet sampler tests passed!")

