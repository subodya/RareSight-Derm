"""
Dataset classes for few-shot episodic learning
Handles N-way K-shot episode construction
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List, Optional, Dict
import medmnist
from medmnist import INFO
import yaml

from src.data.preprocessing import DermaMNISTPreprocessor, ClassBalancedSampler


class EpisodicDermaMNIST(Dataset):
    """
    Episodic dataset for few-shot meta-learning
    Samples N-way K-shot episodes from DermaMNIST
    """
    
    def __init__(
        self,
        split: str = "train",
        config_path: str = "configs/config.yaml",
        download: bool = True
    ):
        """
        Args:
            split: "train", "val", or "test"
            config_path: Path to configuration file
            download: Whether to download DermaMNIST if not present
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
        
        self.split = split
        
        data_flag = 'dermamnist'
        download_flag = download
        
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
        print(f"  Using data root: {root_dir}")
        
        DataClass = getattr(medmnist, INFO[data_flag]['python_class'])
        self.dataset = DataClass(
            split=split,
            download=download_flag,
            root=root_dir
        )
        
        self.preprocessor = DermaMNISTPreprocessor(config_full_path)
        
        self.images = self.dataset.imgs  # Shape: (N, 28, 28, 3)
        self.labels = self.dataset.labels.flatten()  # Shape: (N,)
        
        self.class_names = self.config['dataset']['class_names']
        
        self.class_to_indices = self._organize_by_class()
        
        self.class_counts = {
            cls: len(indices)
            for cls, indices in self.class_to_indices.items()
        }
        
        # Initialize class-balanced sampler (for training only)
        if split == "train":
            self.class_sampler = ClassBalancedSampler(
                class_counts=self.class_counts,
                minority_threshold=200,
                minority_prob=0.4
            )
        
        print(f"\nLoaded DermaMNIST {split} set:")
        print(f"  Total samples: {len(self.images)}")
        print(f"  Number of classes: {len(self.class_to_indices)}")
        print(f"  Class distribution:")
        for cls_id in sorted(self.class_to_indices.keys()):
            print(f"    {cls_id} - {self.class_names[cls_id]}: {self.class_counts[cls_id]} samples")
    
    def _organize_by_class(self) -> Dict[int, List[int]]:
        """Organize dataset indices by class for efficient episode sampling"""
        class_to_indices = {}
        for idx, label in enumerate(self.labels):
            label = int(label)
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)
        return class_to_indices
    
    def sample_episode(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample one N-way K-shot episode
        
        Args:
            n_way: Number of classes in episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
        
        Returns:
            support_images: (N*K, 3, 224, 224)
            support_labels: (N*K,)
            query_images: (N*Q, 3, 224, 224)
            query_labels: (N*Q,)
        """
        # Sample N classes
        if self.split == "train":
            episode_classes = self.class_sampler.sample_classes(n_way)
        else:
            # For val/test, sample uniformly
            available_classes = list(self.class_to_indices.keys())
            episode_classes = np.random.choice(
                available_classes,
                size=n_way,
                replace=False
            ).tolist()
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        # Sample K+Q examples for each class
        for class_idx in episode_classes:
            class_indices = self.class_to_indices[class_idx]
            
            n_samples = k_shot + n_query
            if len(class_indices) < n_samples:
                sampled_indices = np.random.choice(
                    class_indices,
                    size=n_samples,
                    replace=True
                )
            else:
                sampled_indices = np.random.choice(
                    class_indices,
                    size=n_samples,
                    replace=False
                )
            
            support_idx = sampled_indices[:k_shot]
            query_idx = sampled_indices[k_shot:]
            
            # Preprocess images
            for idx in support_idx:
                img = self.images[idx]
                tensor = self.preprocessor.preprocess_image(img)
                support_images.append(tensor)
                support_labels.append(class_idx)
            
            for idx in query_idx:
                img = self.images[idx]
                tensor = self.preprocessor.preprocess_image(img)
                query_images.append(tensor)
                query_labels.append(class_idx)
        
        support_images = torch.stack(support_images)  # (N*K, 3, 224, 224)
        support_labels = torch.tensor(support_labels, dtype=torch.long)  # (N*K,)
        query_images = torch.stack(query_images)  # (N*Q, 3, 224, 224)
        query_labels = torch.tensor(query_labels, dtype=torch.long)  # (N*Q,)
        
        return support_images, support_labels, query_images, query_labels
    
    def __len__(self) -> int:
        """Return number of episodes (arbitrary large number for training)"""
        if self.split == "train":
            return self.config['few_shot']['episodes']
        else:
            return self.config['evaluation']['n_episodes']
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Sample one episode
        
        Args:
            idx: Episode index (ignored, episodes are sampled randomly)
        
        Returns:
            Tuple of (support_images, support_labels, query_images, query_labels)
        """
        n_way = self.config['few_shot']['n_way']
        k_shot = self.config['few_shot']['k_shot']
        n_query = self.config['few_shot']['n_query']
        
        return self.sample_episode(n_way, k_shot, n_query)


if __name__ == "__main__":
    print("Testing EpisodicDermaMNIST dataset...\n")
    
    dataset = EpisodicDermaMNIST(split="train", download=True)
    
    support_img, support_lbl, query_img, query_lbl = dataset.sample_episode(
        n_way=5,
        k_shot=5,
        n_query=15
    )
    
    print(f"\nEpisode shapes:")
    print(f"  Support images: {support_img.shape}")
    print(f"  Support labels: {support_lbl.shape}")
    print(f"  Query images: {query_img.shape}")
    print(f"  Query labels: {query_lbl.shape}")
    
    print(f"\nSupport labels: {support_lbl}")
    print(f"Query labels: {query_lbl}")
    
    print("\nDataset test successful!")