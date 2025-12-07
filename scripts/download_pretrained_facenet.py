"""
Download pretrained FaceNet model from facenet-pytorch library.
"""

import sys
from pathlib import Path
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def download_pretrained_facenet():
    """Download pretrained FaceNet model and save it."""
    try:
        from facenet_pytorch import InceptionResnetV1
        print("facenet-pytorch library found")
    except ImportError:
        print("Installing facenet-pytorch...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "facenet-pytorch"])
        from facenet_pytorch import InceptionResnetV1
        print("facenet-pytorch installed")
    
    # Create models directory
    model_dir = Path(project_root) / "data" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nDownloading pretrained FaceNet model (VGGFace2)...")
    print("This may take a few minutes...")
    
    # Load pretrained model (trained on VGGFace2)
    model = InceptionResnetV1(pretrained='vggface2').eval()
    
    # Get the state dict
    state_dict = model.state_dict()
    
    # Create a checkpoint compatible with our format
    checkpoint = {
        'model_state_dict': state_dict,
        'config': {
            'model_name': 'facenet',
            'embedding_size': 512,
            'pretrained': True,
            'pretrained_source': 'vggface2',
        },
        'epoch': 0,
        'loss': 0.0,
        'metrics': {},
        'note': 'Pretrained FaceNet from facenet-pytorch (VGGFace2)'
    }
    
    # Save checkpoint
    output_path = model_dir / "best_model_facenet_pretrained.pth"
    torch.save(checkpoint, output_path)
    
    print(f"\nPretrained FaceNet model saved to: {output_path}")
    print(f"  Model: InceptionResnetV1")
    print(f"  Pretrained on: VGGFace2")
    print(f"  Embedding size: 512")
    
    # Test the model
    print("\nTesting model...")
    dummy_input = torch.randn(1, 3, 160, 160)
    with torch.no_grad():
        embedding = model(dummy_input)
    print(f"  Input shape: {dummy_input.shape}")
    print(f"  Embedding shape: {embedding.shape}")
    print(f"  Embedding norm: {embedding.norm().item():.4f} (should be ~1.0)")
    
    return str(output_path)


if __name__ == "__main__":
    download_pretrained_facenet()

