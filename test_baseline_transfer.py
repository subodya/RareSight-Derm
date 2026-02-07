"""
Test script for Transfer Learning baseline
Verifies model can run before full training
"""

import torch
from src.models.baseline_transfer import TransferLearningBaseline
from src.data.standard_dataset import StandardDermaMNIST
from torch.utils.data import DataLoader


def test_baseline():
    print("="*60)
    print("TESTING BASELINE 1: TRANSFER LEARNING")
    print("="*60)
    
    # Test 1: Model creation
    print("\n[Test 1] Creating model...")
    model = TransferLearningBaseline(
        num_classes=7,
        pretrained=True,
        freeze_backbone=False
    )
    print(f"✓ Model created: {model.count_parameters():,} trainable parameters")
    
    # Test 2: Forward pass
    print("\n[Test 2] Testing forward pass...")
    dummy_input = torch.randn(8, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (8, 7), f"Expected (8, 7), got {output.shape}"
    print(f"✓ Forward pass successful: {dummy_input.shape} → {output.shape}")
    
    # Test 3: Feature extraction
    print("\n[Test 3] Testing feature extraction...")
    features = model.get_features(dummy_input)
    assert features.shape == (8, 2048), f"Expected (8, 2048), got {features.shape}"
    print(f"✓ Feature extraction successful: {features.shape}")
    
    # Test 4: Dataset loading
    print("\n[Test 4] Loading dataset...")
    train_dataset = StandardDermaMNIST(split='train', download=True)
    print(f"✓ Dataset loaded: {len(train_dataset)} samples")
    
    # Test 5: DataLoader
    print("\n[Test 5] Creating DataLoader...")
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    images, labels = next(iter(train_loader))
    print(f"✓ DataLoader working: batch shape {images.shape}, labels {labels.shape}")
    
    # Test 6: Training step
    print("\n[Test 6] Testing training step...")
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    print(f"✓ Training step successful: loss = {loss.item():.4f}")
    
    # Test 7: Evaluation step
    print("\n[Test 7] Testing evaluation step...")
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
        accuracy = predicted.eq(labels).sum().item() / labels.size(0) * 100
    
    print(f"✓ Evaluation step successful: accuracy = {accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou're ready to run full training:")
    print("  python src/training/train_baseline_transfer.py")
    
    return True


if __name__ == "__main__":
    test_baseline()