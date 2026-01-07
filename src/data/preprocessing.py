"""
Preprocessing utilities for RareSight-Derm
Handles DermaMNIST data loading, resizing, normalization
"""
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Optional
import yaml
from pathlib import Path


class DermaMNISTPreprocessor:
    """
    Preprocesses DermaMNIST images for BiomedCLIP
    - Resize 28×28 → 224×224
    - Handle both grayscale and RGB images
    - Normalize with ImageNet statistics
    """
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        """
        Args:
            config_path: Path to configuration file
        """
        if not os.path.isabs(config_path):
            if os.path.exists(config_path):
                config_full_path = config_path
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                config_full_path = os.path.join(project_root, config_path)
                
                if not os.path.exists(config_full_path):
                    raise FileNotFoundError(
                        f"Config file not found at {config_path} or {config_full_path}. "
                        f"Current directory: {os.getcwd()}"
                    )
        else:
            config_full_path = config_path
        
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
                
            self.original_size = self.config['dataset']['original_size']
            self.target_size = self.config['dataset']['target_size']
            self.mean = self.config['dataset']['mean']
            self.std = self.config['dataset']['std']
            
            self.transform = self._build_transform()
        
    def _build_transform(self) -> transforms.Compose:
        """Build torchvision transform pipeline"""
        return transforms.Compose([
            #Resize from 28×28 to 224×224
            transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std)
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single DermaMNIST image
        
        Args:
            image: NumPy array of shape (28, 28), (28, 28, 1), or (28, 28, 3)
        
        Returns:
            Preprocessed tensor of shape (3, 224, 224)
        """
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
            image_rgb = np.repeat(image, 3, axis=2)
        elif image.ndim == 3:
            if image.shape[2] == 1:
                image_rgb = np.repeat(image, 3, axis=2)
            elif image.shape[2] == 3:
                image_rgb = image
            else:
                raise ValueError(f"Unexpected number of channels: {image.shape[2]}")
        else:
            raise ValueError(f"Unexpected image dimensions: {image.ndim}")
        
        # Ensure uint8 type for PIL
        if image_rgb.dtype != np.uint8:
            if image_rgb.max() <= 1.0:
                image_rgb = (image_rgb * 255).astype(np.uint8)
            else:
                image_rgb = image_rgb.astype(np.uint8)
        
        pil_image = Image.fromarray(image_rgb)
        
        tensor = self.transform(pil_image)
        
        return tensor
    
    def preprocess_batch(self, images: np.ndarray) -> torch.Tensor:
        """
        Preprocess a batch of images
        
        Args:
            images: NumPy array of shape (B, 28, 28) or (B, 28, 28, 1) or (B, 28, 28, 3)
        
        Returns:
            Batch tensor of shape (B, 3, 224, 224)
        """
        batch_tensors = []
        for img in images:
            tensor = self.preprocess_image(img)
            batch_tensors.append(tensor)
        
        return torch.stack(batch_tensors, dim=0)
    
    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        """
        Reverse normalization for visualization
        
        Args:
            tensor: Normalized tensor of shape (3, 224, 224)
        
        Returns:
            Denormalized array for visualization
        """
        denorm = tensor.clone()
        
        for t, m, s in zip(denorm, self.mean, self.std):
            t.mul_(s).add_(m)
        
        denorm = torch.clamp(denorm, 0, 1)
        
        return denorm.permute(1, 2, 0).cpu().numpy()


class ClassBalancedSampler:
    """
    Samples episodes with balanced representation of minority classes
    Critical for handling DermaMNIST's class imbalance
    """
    
    def __init__(
        self,
        class_counts: dict,
        minority_threshold: int = 200,
        minority_prob: float = 0.4
    ):
        """
        Args:
            class_counts: Dictionary mapping class_id → number of samples
            minority_threshold: Classes with < threshold samples are "minority"
            minority_prob: Probability of including minority class in episode
        """
        self.class_counts = class_counts
        self.minority_threshold = minority_threshold
        self.minority_prob = minority_prob
        
        # Categorizing classes
        self.minority_classes = [
            cls for cls, count in class_counts.items()
            if count < minority_threshold
        ]
        self.majority_classes = [
            cls for cls, count in class_counts.items()
            if count >= minority_threshold
        ]
        
        print(f"Minority classes (<{minority_threshold} samples): {self.minority_classes}")
        print(f"Majority classes (≥{minority_threshold} samples): {self.majority_classes}")
    
    def sample_classes(self, n_way: int) -> List[int]:
        """
        Sample N classes with balanced minority representation
        
        Args:
            n_way: Number of classes to sample
        
        Returns:
            List of class IDs
        """
        include_minority = (
            len(self.minority_classes) > 0 and
            np.random.rand() < self.minority_prob
        )
        
        if include_minority and n_way > 1:
            n_minority = min(
                np.random.randint(1, 3),
                len(self.minority_classes),
                n_way
            )
            minority_sample = np.random.choice(
                self.minority_classes,
                size=n_minority,
                replace=False
            )
            
            n_majority = n_way - n_minority
            majority_sample = np.random.choice(
                self.majority_classes,
                size=n_majority,
                replace=False
            )
            
            sampled_classes = np.concatenate([minority_sample, majority_sample])
        else:
            all_classes = self.minority_classes + self.majority_classes
            sampled_classes = np.random.choice(
                all_classes,
                size=n_way,
                replace=False
            )
        
        return sampled_classes.tolist()


def split_meta_train_test(
    dataset,
    meta_train_classes: List[int],
    meta_test_classes: List[int]
) -> Tuple:
    """
    Split dataset into meta-training and meta-testing sets
    
    Args:
        dataset: MedMNIST dataset object
        meta_train_classes: List of class indices for training
        meta_test_classes: List of class indices for testing
    
    Returns:
        (train_data, train_labels), (test_data, test_labels)
    """
    images = dataset.imgs  # Shape: (N, 28, 28, 3) or (N, 28, 28, 1)
    labels = dataset.labels.flatten()  # Shape: (N,)
    
    # Create masks
    train_mask = np.isin(labels, meta_train_classes)
    test_mask = np.isin(labels, meta_test_classes)
    
    # Split data
    train_data = images[train_mask]
    train_labels = labels[train_mask]
    test_data = images[test_mask]
    test_labels = labels[test_mask]
    
    print(f"\nMeta-Training Set:")
    print(f"  Classes: {meta_train_classes}")
    print(f"  Samples: {len(train_data)}")
    for cls in meta_train_classes:
        count = np.sum(train_labels == cls)
        print(f"    Class {cls}: {count} samples")
    
    print(f"\nMeta-Testing Set:")
    print(f"  Classes: {meta_test_classes}")
    print(f"  Samples: {len(test_data)}")
    for cls in meta_test_classes:
        count = np.sum(test_labels == cls)
        print(f"    Class {cls}: {count} samples")
    
    return (train_data, train_labels), (test_data, test_labels)


if __name__ == "__main__":
    # Test preprocessing
    preprocessor = DermaMNISTPreprocessor()
    
    # Test with RGB image (28, 28, 3)
    print("Testing RGB image (28, 28, 3):")
    dummy_image_rgb = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
    preprocessed_rgb = preprocessor.preprocess_image(dummy_image_rgb)
    print(f"  Input shape: {dummy_image_rgb.shape}")
    print(f"  Output shape: {preprocessed_rgb.shape}")
    print(f"  Output range: [{preprocessed_rgb.min():.3f}, {preprocessed_rgb.max():.3f}]")
    
    # Test with grayscale image (28, 28, 1)
    print("\nTesting grayscale image (28, 28, 1):")
    dummy_image_gray = np.random.randint(0, 256, (28, 28, 1), dtype=np.uint8)
    preprocessed_gray = preprocessor.preprocess_image(dummy_image_gray)
    print(f"  Input shape: {dummy_image_gray.shape}")
    print(f"  Output shape: {preprocessed_gray.shape}")
    print(f"  Output range: [{preprocessed_gray.min():.3f}, {preprocessed_gray.max():.3f}]")
    
    # Test with 2D grayscale (28, 28)
    print("\nTesting 2D grayscale image (28, 28):")
    dummy_image_2d = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
    preprocessed_2d = preprocessor.preprocess_image(dummy_image_2d)
    print(f"  Input shape: {dummy_image_2d.shape}")
    print(f"  Output shape: {preprocessed_2d.shape}")
    print(f"  Output range: [{preprocessed_2d.min():.3f}, {preprocessed_2d.max():.3f}]")
    
    print("\nAll preprocessing tests successful! ✓")