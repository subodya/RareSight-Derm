"""
Standard dataset for supervised learning baselines
Returns individual (image, label) pairs
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, Optional
import medmnist
from medmnist import INFO
import yaml

from src.data.preprocessing import DermaMNISTPreprocessor


class StandardDermaMNIST(Dataset):
    """
    Standard dataset for supervised learning
    Returns (image, label) pairs for batch training
    """
    
    def __init__(
        self,
        split: str = "train",
        config_path: str = "configs/config.yaml",
        download: bool = True,
        augment: bool = False
    ):
        """
        Args:
            split: "train", "val", or "test"
            config_path: Path to configuration file
            download: Whether to download DermaMNIST
            augment: Apply data augmentation (training only)
        """
        # Load config
        if not os.path.isabs(config_path):
            if os.path.exists(config_path):
                config_full_path = config_path
            else:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                project_root = os.path.dirname(os.path.dirname(current_dir))
                config_full_path = os.path.join(project_root, config_path)
        else:
            config_full_path = config_path
        
        with open(config_full_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.split = split
        self.augment = augment and (split == "train")
        
        # Load DermaMNIST
        data_flag = 'dermamnist'
        
        # Handle root directory
        root_dir_config = self.config['dataset']['root_dir']
        if root_dir_config is None or root_dir_config == "":
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            root_dir = os.path.join(project_root, "data", "raw")
        elif not os.path.isabs(root_dir_config):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            root_dir = os.path.join(project_root, root_dir_config)
        else:
            root_dir = root_dir_config
        
        os.makedirs(root_dir, exist_ok=True)
        
        DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
        self.dataset = DataClass(
            split=split,
            download=download,
            root=root_dir
        )
        
        # Initialize preprocessor
        self.preprocessor = DermaMNISTPreprocessor(config_full_path)
        
        # Extract images and labels
        self.images = self.dataset.imgs  # (N, 28, 28, 3)
        self.labels = self.dataset.labels.flatten()  # (N,)
        
        # Class names
        self.class_names = self.config['dataset']['class_names']
        
        # Count samples per class
        unique, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique, counts))
        
        print(f"\nLoaded DermaMNIST {split} set (Standard):")
        print(f"  Total samples: {len(self.images)}")
        print(f"  Class distribution:")
        for cls_id in sorted(self.class_counts.keys()):
            print(f"    {cls_id}: {self.class_counts[cls_id]} samples")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get one (image, label) pair
        
        Args:
            idx: Index
        
        Returns:
            image: Preprocessed tensor (3, 224, 224)
            label: Class label (int)
        """
        # Get image and label
        image = self.images[idx]
        label = int(self.labels[idx])
        
        # Preprocess
        image_tensor = self.preprocessor.preprocess_image(image)
        
        # TODO: Add augmentation if self.augment
        # (Will implement in training script with torchvision transforms)
        
        return image_tensor, label


# Test the dataset
if __name__ == "__main__":
    print("Testing StandardDermaMNIST dataset...\n")
    
    # Create dataset
    train_dataset = StandardDermaMNIST(split="train", download=True)
    val_dataset = StandardDermaMNIST(split="val", download=True)
    test_dataset = StandardDermaMNIST(split="test", download=True)
    
    # Test __getitem__
    img, label = train_dataset[0]
    print(f"\nSample data point:")
    print(f"  Image shape: {img.shape}")
    print(f"  Label: {label}")
    print(f"  Label type: {type(label)}")
    
    # Create DataLoader
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0  # Set to 4 for actual training
    )
    
    # Test batch
    batch_img, batch_label = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Images: {batch_img.shape}")
    print(f"  Labels: {batch_label.shape}")
    
    print("\nDataset test successful!")