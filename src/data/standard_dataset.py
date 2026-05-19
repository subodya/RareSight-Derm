"""
Standard dataset for supervised learning baselines.
Returns individual (image, label) pairs for batch training.
Supports HAM10000 (primary) and DermaMNIST (legacy fallback).
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import yaml

from src.data.preprocessing import (
    DermaMNISTPreprocessor,
    load_ham10000,
    HAM10000_CLASS_NAMES,
    _resolve_config,
)


class StandardDermaMNIST(Dataset):
    """
    Standard supervised-learning dataset.
    Loads HAM10000 by default; falls back to DermaMNIST when
    config['dataset']['name'] == 'dermamnist'.
    """

    def __init__(
        self,
        split: str = "train",
        config_path: str = "configs/config.yaml",
        download: bool = True,
        augment: bool = False,
    ):
        config_full_path = _resolve_config(config_path)
        with open(config_full_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.split = split
        self.augment = augment and (split == "train")
        self.preprocessor = DermaMNISTPreprocessor(config_full_path)

        dataset_name = self.config["dataset"].get("name", "ham10000").lower()

        if dataset_name == "ham10000":
            self._load_ham10000()
        else:
            self._load_dermamnist(download)

        unique, counts = np.unique(self.labels, return_counts=True)
        self.class_counts = dict(zip(unique.tolist(), counts.tolist()))

        print(f"\nLoaded {'HAM10000' if self._use_paths else 'DermaMNIST'} [{split}] (Standard):")
        print(f"  Total samples : {len(self)}")
        for cls_id in sorted(self.class_counts):
            name = self.class_names.get(cls_id, str(cls_id))
            print(f"  Class {cls_id} ({name}): {self.class_counts[cls_id]}")

    # ------------------------------------------------------------------ #
    # Loaders                                                              #
    # ------------------------------------------------------------------ #

    def _load_ham10000(self) -> None:
        self._use_paths = True

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))

        ham_dir = self.config["dataset"].get("ham10000_dir", "data/ham10000")
        if not os.path.isabs(ham_dir):
            ham_dir = os.path.join(project_root, ham_dir)

        self.image_paths, self.labels = load_ham10000(
            data_root=ham_dir,
            split=self.split,
            val_size=self.config["dataset"].get("val_size", 0.1),
            test_size=self.config["dataset"].get("test_size", 0.1),
            seed=self.config["dataset"].get("split_seed", 42),
        )
        self.class_names = HAM10000_CLASS_NAMES

    def _load_dermamnist(self, download: bool) -> None:
        import medmnist
        from medmnist import INFO

        self._use_paths = False

        root_dir = self.config["dataset"].get("root_dir") or ""
        if not root_dir:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            root_dir = os.path.join(project_root, "data", "raw")
        elif not os.path.isabs(root_dir):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            root_dir = os.path.join(project_root, root_dir)

        os.makedirs(root_dir, exist_ok=True)
        DataClass = getattr(medmnist, INFO["dermamnist"]["python_class"])
        dataset = DataClass(split=self.split, download=download, root=root_dir)

        self.images = dataset.imgs
        self.labels = dataset.labels.flatten()
        self.class_names = self.config["dataset"].get("class_names", {})

    # ------------------------------------------------------------------ #
    # Dataset protocol                                                     #
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.image_paths) if self._use_paths else len(self.images)

    def __getitem__(self, idx: int):
        label = int(self.labels[idx])

        if self._use_paths:
            pil = Image.open(self.image_paths[idx]).convert("RGB")
            image_tensor = self.preprocessor.preprocess_pil(pil, augment=self.augment)
        else:
            image_tensor = self.preprocessor.preprocess_image(self.images[idx], augment=self.augment)

        return image_tensor, label

    # Convenience: class weights for focal / weighted cross-entropy loss
    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency class weights for imbalanced training."""
        n_classes = len(self.class_counts)
        total = sum(self.class_counts.values())
        weights = torch.zeros(n_classes)
        for cls_id, count in self.class_counts.items():
            weights[cls_id] = total / (n_classes * count)
        return weights / weights.sum() * n_classes  # normalise so sum = n_classes
