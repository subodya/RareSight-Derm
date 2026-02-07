"""
Test script for Standard ProtoNet baseline
"""

import torch
from src.models.baseline_protonet import StandardProtoNet
from src.data.dataset import EpisodicDermaMNIST


def test_protonet():
    print("="*60)
    print("TESTING BASELINE 2: STANDARD PROTONET")
    print("="*60)
    
    # Test 1: Model creation
    print("\n[Test 1] Creating model...")
    model = StandardProtoNet(
        backbone="resnet50",
        pretrained=True,
        freeze_backbone=False,
        feature_dim=512,
        distance="euclidean"
    )
    print(f"✓ Model created")
    
    # Test 2: Feature extraction
    print("\n[Test 2] Testing feature extraction...")
    dummy_images = torch.randn(10, 3, 224, 224)
    features = model(dummy_images)
    assert features.shape == (10, 512), f"Expected (10, 512), got {features.shape}"
    print(f"✓ Feature extraction: {dummy_images.shape} → {features.shape}")
    
    # Test 3: Episode sampling
    print("\n[Test 3] Loading episodic dataset...")
    dataset = EpisodicDermaMNIST(split='train', download=True)
    
    support_img, support_lbl, query_img, query_lbl = dataset.sample_episode(
        n_way=5, k_shot=5, n_query=15
    )
    print(f"✓ Episode sampled:")
    print(f"  Support: {support_img.shape}, labels: {support_lbl.shape}")
    print(f"  Query: {query_img.shape}, labels: {query_lbl.shape}")
    
    # Test 4: Prototype computation
    print("\n[Test 4] Testing prototype computation...")
    model.eval()
    with torch.no_grad():
        support_features = model(support_img)
        prototypes = model.compute_prototypes(support_features, support_lbl, n_way=5)
    assert prototypes.shape == (5, 512), f"Expected (5, 512), got {prototypes.shape}"
    print(f"✓ Prototypes computed: {prototypes.shape}")
    
    # Test 5: Distance computation
    print("\n[Test 5] Testing distance computation...")
    with torch.no_grad():
        query_features = model(query_img)
        distances = model.compute_distances(query_features, prototypes)
    assert distances.shape == (75, 5), f"Expected (75, 5), got {distances.shape}"
    print(f"✓ Distances computed: {distances.shape}")
    
    # Test 6: Loss computation
    print("\n[Test 6] Testing loss computation...")
    model.train()
    loss, accuracy = model.loss(
        support_img, support_lbl,
        query_img, query_lbl,
        n_way=5
    )
    print(f"✓ Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
    
    # Test 7: Backward pass
    print("\n[Test 7] Testing backward pass...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss.backward()
    optimizer.step()
    print(f"✓ Backward pass successful")
    
    # Test 8: Prediction
    print("\n[Test 8] Testing prediction...")
    model.eval()
    predictions = model.predict(support_img, support_lbl, query_img, n_way=5)
    assert predictions.shape == (75,), f"Expected (75,), got {predictions.shape}"
    pred_accuracy = predictions.eq(query_lbl).float().mean().item() * 100
    print(f"✓ Predictions: {predictions.shape}, Accuracy: {pred_accuracy:.2f}%")
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nYou're ready to run full training:")
    print("  python src/training/train_baseline_protonet.py")
    
    return True


if __name__ == "__main__":
    test_protonet()