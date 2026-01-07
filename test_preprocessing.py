
import sys
import torch
import numpy as np
from src.data.dataset import EpisodicDermaMNIST
from src.data.preprocessing import DermaMNISTPreprocessor

def test_preprocessing():
    print("=" * 60)
    print("RARESIGHT-DERM PREPROCESSING TEST")
    print("=" * 60)
    
    print("\n[Test 1] Checking hardware...")
    if torch.cuda.is_available():
        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  PyTorch version: {torch.__version__}")
    else:
        print("⚠ CUDA not available, will use CPU")
    
    print("\n[Test 2] Loading DermaMNIST dataset...")
    try:
        dataset = EpisodicDermaMNIST(split="train", download=True)
        print(f"✓ Dataset loaded: {len(dataset.images)} images")
        print(f"  Classes: {len(dataset.class_to_indices)}")
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        return False
    
    print("\n[Test 3] Testing preprocessor...")
    try:
        preprocessor = DermaMNISTPreprocessor()
        
        test_img = dataset.images[0]
        print(f"  Input shape: {test_img.shape}")
        
        processed = preprocessor.preprocess_image(test_img)
        print(f"  Output shape: {processed.shape}")
        print(f"  Output dtype: {processed.dtype}")
        print(f"  Output range: [{processed.min():.3f}, {processed.max():.3f}]")
        
        assert processed.shape == (3, 224, 224), "Unexpected output shape!"
        print("✓ Preprocessing successful")
        
    except Exception as e:
        print(f"✗ Preprocessing failed: {e}")
        return False
    
    print("\n[Test 4] Sampling episode...")
    try:
        support_img, support_lbl, query_img, query_lbl = dataset.sample_episode(
            n_way=5, k_shot=5, n_query=15
        )
        
        print(f"  Support images: {support_img.shape}")
        print(f"  Support labels: {support_lbl.shape}")
        print(f"  Query images: {query_img.shape}")
        print(f"  Query labels: {query_lbl.shape}")
        print(f"  Episode classes: {np.unique(support_lbl.numpy())}")
        
        assert support_img.shape == (25, 3, 224, 224), "Unexpected support shape!"
        assert query_img.shape == (75, 3, 224, 224), "Unexpected query shape!"
        print("✓ Episode sampling successful")
        
    except Exception as e:
        print(f"✗ Episode sampling failed: {e}")
        return False
    
    print("\n[Test 5] Checking class distribution...")
    try:
        print("  Class counts:")
        for cls_id in sorted(dataset.class_counts.keys()):
            count = dataset.class_counts[cls_id]
            name = dataset.class_names[cls_id]
            marker = "⚠ MINORITY" if count < 200 else ""
            print(f"    Class {cls_id}: {count:4d} samples - {name[:30]}... {marker}")
        
        print("✓ Class distribution checked")
        
    except Exception as e:
        print(f"✗ Class distribution check failed: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED!")
    print("=" * 60)

    return True

if __name__ == "__main__":
    success = test_preprocessing()
    sys.exit(0 if success else 1)