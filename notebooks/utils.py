"""
Utility functions for the face recognition demo notebook.
All helper functions are here to keep the notebook clean and simple.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import urllib.request


def load_face_detector():
    """Load OpenCV DNN face detector model."""
    model_dir = Path("../data/models/face_detection")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    prototxt_path = model_dir / "deploy.prototxt"
    model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    
    if not prototxt_path.exists():
        print("Downloading face detector prototxt...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            str(prototxt_path)
        )
    
    if not model_path.exists():
        print("Downloading face detector model (this may take a moment)...")
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            str(model_path)
        )
    
    net = cv2.dnn.readNetFromCaffe(str(prototxt_path), str(model_path))
    return net


def resolve_image_path(image_path):
    """Resolve image path relative to project root."""
    image_path = Path(image_path)
    if not image_path.is_absolute():
        project_root = Path().absolute().parent
        image_path = project_root / image_path
    return str(image_path.resolve())


def load_image(image_path, heif_supported=False):
    """Load image, handling HEIF and other formats."""
    image_path_str = resolve_image_path(image_path)
    
    # Try OpenCV first
    image_bgr = cv2.imread(image_path_str)
    
    if image_bgr is None:
        # Try PIL for HEIF/other formats
        try:
            if heif_supported:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            pil_image = Image.open(image_path_str).convert('RGB')
            image = np.array(pil_image)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            raise FileNotFoundError(f"Could not load image: {e}")
    
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image, image_path_str


def detect_faces(image, detector, confidence_threshold=0.5):
    """Detect faces in image using OpenCV DNN."""
    h, w = image.shape[:2]
    
    # Prepare blob
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(
        cv2.resize(image_bgr, (300, 300)), 
        1.0, 
        (300, 300), 
        [104, 117, 123]
    )
    
    # Detect
    detector.setInput(blob)
    detections = detector.forward()
    
    # Process detections
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            width = x2 - x
            height = y2 - y
            faces.append({
                'box': [x, y, width, height],
                'confidence': confidence
            })
    
    return faces


def visualize_face_detection(image, faces):
    """Visualize face detection results."""
    image_with_boxes = image.copy()
    
    for i, face in enumerate(faces, 1):
        x, y, w, h = face['box']
        confidence = face['confidence']
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image_with_boxes, f"Face {i}: {confidence:.2%}", 
                   (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(image)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    axes[1].imshow(image_with_boxes)
    axes[1].set_title(f'Face Detection ({len(faces)} face(s) found)', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    plt.tight_layout()
    plt.show()
    
    if len(faces) > 0:
        print(f"Found {len(faces)} face(s)!")
        for i, face in enumerate(faces, 1):
            print(f"  Face {i}: Confidence {face['confidence']:.2%}, Box: {face['box']}")
    else:
        print("No faces detected!")


def visualize_preprocessing(image_path):
    """Visualize preprocessing pipeline."""
    image_path_str = resolve_image_path(image_path)
    original = Image.open(image_path_str).convert('RGB')
    
    # Apply transforms step by step
    resized = original.resize((224, 224), Image.Resampling.LANCZOS)
    
    from torchvision import transforms
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(resized)
    
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    normalized = (tensor - mean) / std
    
    def tensor_to_image(t, denormalize=True):
        if denormalize:
            t = t * std + mean
        t = torch.clamp(t, 0, 1)
        t = t.permute(1, 2, 0)
        return (t.numpy() * 255).astype(np.uint8)
    
    # Visualize
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    axes[0, 0].imshow(original)
    axes[0, 0].set_title(f'1. Original Image\nSize: {original.size[0]}x{original.size[1]}', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(resized)
    axes[0, 1].set_title('2. Resized to 224x224', fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    axes[1, 0].imshow(tensor_to_image(tensor, denormalize=False))
    axes[1, 0].set_title('3. Converted to Tensor\n(0-1 range)', fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    axes[1, 1].imshow(tensor_to_image(normalized, denormalize=True))
    axes[1, 1].set_title('4. Normalized\n(ImageNet stats)', fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    plt.tight_layout()
    plt.show()
    
    print("\nPreprocessing Statistics:")
    print(f"  Original size: {original.size}")
    print(f"  Resized size: {resized.size}")
    print(f"  Tensor shape: {tensor.shape}")
    print(f"  Tensor range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    print(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")


def visualize_embedding(embedding, image_path, title="Face Embedding"):
    """Visualize face embedding."""
    embedding_np = embedding.squeeze().numpy()
    image_path_str = resolve_image_path(image_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    img = Image.open(image_path_str).convert('RGB')
    axes[0].imshow(img)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].bar(range(len(embedding_np)), embedding_np)
    axes[1].set_title(f'{title}\n128-dimensional embedding vector', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Dimension', fontsize=12)
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print(f"\nEmbedding Statistics:")
    print(f"  Shape: {embedding_np.shape}")
    print(f"  Min: {embedding_np.min():.4f}")
    print(f"  Max: {embedding_np.max():.4f}")
    print(f"  Mean: {embedding_np.mean():.4f}")
    print(f"  Std: {embedding_np.std():.4f}")
    print(f"  Norm (L2): {np.linalg.norm(embedding_np):.4f} (should be ~1.0)")


def visualize_comparison(reference_path, test_path, embedding1, embedding2, distance, is_match, threshold, confidence):
    """Visualize face comparison results."""
    ref_path = resolve_image_path(reference_path)
    test_path_str = resolve_image_path(test_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    
    img1 = Image.open(ref_path).convert('RGB')
    img2 = Image.open(test_path_str).convert('RGB')
    
    axes[0, 0].imshow(img1)
    axes[0, 0].set_title('Reference Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    if is_match:
        title = f'Test Image\nMATCH! (Same Person)'
        color = 'green'
    else:
        title = f'Test Image\nNO MATCH (Different Person)'
        color = 'red'
    axes[0, 1].imshow(img2)
    axes[0, 1].set_title(title, fontsize=14, fontweight='bold', color=color)
    axes[0, 1].axis('off')
    
    # Embedding comparison
    emb1_np = embedding1.squeeze().numpy()
    emb2_np = embedding2.squeeze().numpy()
    
    x = np.arange(len(emb1_np))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, emb1_np, width, label='Reference', alpha=0.7)
    axes[1, 0].bar(x + width/2, emb2_np, width, label='Test', alpha=0.7)
    axes[1, 0].set_title('Embedding Comparison', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Dimension', fontsize=12)
    axes[1, 0].set_ylabel('Value', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Distance visualization
    axes[1, 1].barh(['Distance'], [distance], color='green' if is_match else 'red', alpha=0.7)
    axes[1, 1].axvline(threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    axes[1, 1].set_title(f'Distance: {distance:.4f}', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Euclidean Distance', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("RECOGNITION RESULTS")
    print("=" * 60)
    print(f"Distance: {distance:.4f}")
    print(f"Threshold: {threshold}")
    print(f"Confidence: {confidence:.2%}")
    print()
    
    if is_match:
        print(f"MATCH! Same person detected.")
    else:
        print(f"NO MATCH. Different person or too different.")

