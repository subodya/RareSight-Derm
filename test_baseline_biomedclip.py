"""
Test script for Zero-Shot BiomedCLIP baseline
Quick check before full evaluation
"""

import torch
from src.models.baseline_biomedclip import ZeroShotBiomedCLIP


def test_biomedclip():
    print("="*60)
    print("TESTING BASELINE 3: ZERO-SHOT BIOMEDCLIP")
    print("="*60)
    
    # Test 1: Model loading
    print("\n[Test 1] Loading BiomedCLIP model...")
    model = ZeroShotBiomedCLIP()
    print("✓ Model loaded successfully")
    
    # Test 2: Image encoding
    print("\n[Test 2] Testing image encoding...")
    dummy_images = torch.randn(4, 3, 224, 224)
    image_features = model.encode_images(dummy_images)
    print(f"✓ Image encoding: {dummy_images.shape} → {image_features.shape}")
    
    # Test 3: Text encoding
    print("\n[Test 3] Testing text encoding...")
    text_descriptions = [
        "A dermoscopy image of melanoma",
        "A dermoscopy image of melanocytic nevi",
        "A dermoscopy image of basal cell carcinoma"
    ]
    text_features = model.encode_texts(text_descriptions)
    print(f"✓ Text encoding: {len(text_descriptions)} texts → {text_features.shape}")
    
    # Test 4: Zero-shot prediction
    print("\n[Test 4] Testing zero-shot prediction...")
    logits, predictions = model.predict(dummy_images, text_descriptions)
    print(f"✓ Zero-shot prediction:")
    print(f"  Logits shape: {logits.shape}")
    print(f"  Predictions: {predictions.tolist()}")
    
    # Test 5: Feature normalization check
    print("\n[Test 5] Checking feature normalization...")
    image_norms = torch.norm(image_features, dim=1)
    text_norms = torch.norm(text_features, dim=1)
    print(f"✓ Image feature norms: {image_norms.tolist()}")
    print(f"✓ Text feature norms: {text_norms.tolist()}")
    assert torch.allclose(image_norms, torch.ones_like(image_norms), atol=1e-5), "Images not normalized!"
    assert torch.allclose(text_norms, torch.ones_like(text_norms), atol=1e-5), "Texts not normalized!"
    print("✓ All features properly normalized")
    
    # Test 6: GPU check
    print("\n[Test 6] Checking device...")
    device = next(model.parameters()).device
    print(f"✓ Model on device: {device}")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou're ready to run full evaluation:")
    print("  python src/training/eval_baseline_biomedclip.py")
    
    return True


if __name__ == "__main__":
    test_biomedclip()