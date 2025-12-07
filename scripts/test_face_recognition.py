#!/usr/bin/env python3
"""
Test face recognition on your own images.
Upload a reference image, then test recognition on another image.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse

from src.models.face_net import create_face_net
from src.models.facenet_pretrained import create_facenet_pretrained
from src.data.lfw_dataset import get_default_transforms
from training.train_utils import load_checkpoint
from training.config import TrainingConfig


class FaceRecognizer:
    """Face recognition system using trained model."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Initialize face recognizer.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load config from checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        config_dict = checkpoint.get('config', {})
        
        # Detect model type from checkpoint
        model_name = config_dict.get('model_name', 'resnet18')  # Default to resnet18 for old checkpoints
        embedding_size = config_dict.get('embedding_size', 128)
        
        # Create appropriate model based on checkpoint
        if model_name == 'facenet':
            print(f"Loading FaceNet model (embedding_size={embedding_size})...")
            self.model = create_facenet_pretrained(
                embedding_size=embedding_size,
                pretrained=False  # We're loading trained weights
            ).to(self.device)
        else:  # resnet18 or default
            print(f"Loading ResNet18 model (embedding_size={embedding_size})...")
            self.model = create_face_net(
                embedding_size=embedding_size,
                pretrained=False  # We're loading trained weights
            ).to(self.device)
        
        # Load trained weights
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        print(f"✓ Loaded model from {model_path}")
        print(f"  Model type: {model_name}")
        print(f"  Embedding size: {embedding_size}")
        if checkpoint.get('epoch') is not None:
            print(f"  Trained for {checkpoint.get('epoch', 'unknown')} epochs")
            print(f"  Best validation loss: {checkpoint.get('loss', 'unknown'):.4f}")
        else:
            print(f"  Note: {checkpoint.get('note', 'Pretrained model')}")
        
        # Setup transforms
        self.transform = get_default_transforms(is_training=False)
        
        # Database of known faces
        self.known_faces = {}  # {name: embedding}
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Preprocessed image tensor
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)  # Add batch dimension
        return image_tensor, image
    
    def get_embedding(self, image_path: str) -> torch.Tensor:
        """
        Get face embedding from an image.
        
        Args:
            image_path: Path to image file
        
        Returns:
            Face embedding vector
        """
        image_tensor, _ = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        with torch.no_grad():
            embedding = self.model(image_tensor)
        
        return embedding.cpu()
    
    def add_known_face(self, name: str, image_path: str):
        """
        Add a known face to the database.
        
        Args:
            name: Name/ID of the person
            image_path: Path to reference image
        """
        print(f"Adding {name} to database...")
        embedding = self.get_embedding(image_path)
        self.known_faces[name] = embedding
        print(f"✓ Added {name}")
    
    def recognize_face(self, image_path: str, threshold: float = 0.6) -> tuple:
        """
        Recognize a face in an image.
        
        Args:
            image_path: Path to image to recognize
            threshold: Distance threshold for recognition (lower = stricter)
        
        Returns:
            (name, distance, confidence) or (None, distance, confidence) if unknown
        """
        embedding = self.get_embedding(image_path)
        
        if len(self.known_faces) == 0:
            return None, None, 0.0
        
        # Calculate distances to all known faces
        distances = {}
        for name, known_embedding in self.known_faces.items():
            # Euclidean distance
            distance = F.pairwise_distance(embedding, known_embedding, p=2).item()
            distances[name] = distance
        
        # Find closest match
        best_match = min(distances.items(), key=lambda x: x[1])
        name, distance = best_match
        
        # Check if distance is below threshold
        if distance < threshold:
            # Confidence: inverse of distance (normalized)
            confidence = max(0, 1 - (distance / threshold))
            return name, distance, confidence
        else:
            return None, distance, 0.0
    
    def visualize_preprocessing(self, image_path: str, save_path: str = None):
        """
        Visualize how the image looks after preprocessing.
        
        Args:
            image_path: Path to original image
            save_path: Optional path to save visualization
        """
        # Load original
        original = Image.open(image_path).convert('RGB')
        
        # Get preprocessed
        processed_tensor, _ = self.preprocess_image(image_path)
        
        # Convert back to PIL for visualization
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        processed_tensor = processed_tensor.squeeze(0) * std + mean
        processed_tensor = torch.clamp(processed_tensor, 0, 1)
        
        # Convert to numpy
        processed_np = processed_tensor.permute(1, 2, 0).numpy()
        processed_np = (processed_np * 255).astype(np.uint8)
        processed = Image.fromarray(processed_np)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image', fontsize=14)
        axes[0].axis('off')
        
        axes[1].imshow(processed)
        axes[1].set_title('After Preprocessing (224x224, normalized)', fontsize=14)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test face recognition on your images')
    parser.add_argument(
        '--model',
        type=str,
        default='data/models/best_model.pth',
        help='Path to trained model checkpoint (default: data/models/best_model.pth)'
    )
    parser.add_argument(
        '--reference',
        type=str,
        required=True,
        help='Path to reference image (your face to enroll)'
    )
    parser.add_argument(
        '--name',
        type=str,
        default='You',
        help='Name/ID for the reference face (default: You)'
    )
    parser.add_argument(
        '--test',
        type=str,
        required=True,
        help='Path to test image (image to recognize)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.6,
        help='Recognition threshold (lower = stricter, default: 0.6)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show preprocessing visualization'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu, default: cuda)'
    )
    
    args = parser.parse_args()
    
    # Check if files exist
    if not Path(args.model).exists():
        print(f"Error: Model file not found: {args.model}")
        print("Available models:")
        for p in Path('data/models').glob('*.pth'):
            print(f"  - {p}")
        return
    
    if not Path(args.reference).exists():
        print(f"Error: Reference image not found: {args.reference}")
        return
    
    if not Path(args.test).exists():
        print(f"Error: Test image not found: {args.test}")
        return
    
    print("=" * 60)
    print("Face Recognition Test")
    print("=" * 60)
    
    # Initialize recognizer
    recognizer = FaceRecognizer(args.model, device=args.device)
    
    # Visualize preprocessing if requested
    if args.visualize:
        print("\n1. Visualizing preprocessing...")
        recognizer.visualize_preprocessing(
            args.reference,
            save_path='reference_preprocessing.png'
        )
        recognizer.visualize_preprocessing(
            args.test,
            save_path='test_preprocessing.png'
        )
    
    # Add reference face
    print(f"\n2. Enrolling reference face: {args.name}")
    recognizer.add_known_face(args.name, args.reference)
    
    # Test recognition
    print(f"\n3. Testing recognition on: {args.test}")
    name, distance, confidence = recognizer.recognize_face(args.test, threshold=args.threshold)
    
    # Results
    print("\n" + "=" * 60)
    print("Recognition Results")
    print("=" * 60)
    print(f"Distance to reference: {distance:.4f}")
    print(f"Threshold: {args.threshold}")
    print(f"Confidence: {confidence:.2%}")
    
    if name:
        print(f"\n✓ RECOGNIZED as: {name}")
        print(f"  Distance: {distance:.4f} (below threshold {args.threshold})")
        print(f"  Confidence: {confidence:.2%}")
    else:
        print(f"\n✗ NOT RECOGNIZED (Unknown face)")
        print(f"  Distance: {distance:.4f} (above threshold {args.threshold})")
        print(f"  Try lowering threshold (current: {args.threshold})")
    
    # Create visualization
    print("\n4. Creating result visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Reference image
    ref_img = Image.open(args.reference).convert('RGB')
    axes[0].imshow(ref_img)
    axes[0].set_title(f'Reference: {args.name}', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Test image
    test_img = Image.open(args.test).convert('RGB')
    axes[1].imshow(test_img)
    if name:
        title = f'Test Image\n✓ RECOGNIZED as {name}\nDistance: {distance:.4f}, Confidence: {confidence:.2%}'
        color = 'green'
    else:
        title = f'Test Image\n✗ NOT RECOGNIZED\nDistance: {distance:.4f} (threshold: {args.threshold})'
        color = 'red'
    axes[1].set_title(title, fontsize=14, fontweight='bold', color=color)
    axes[1].axis('off')
    
    plt.tight_layout()
    output_path = 'recognition_result.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved result visualization to {output_path}")
    plt.close()
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

