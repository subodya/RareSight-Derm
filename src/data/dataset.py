"""
Episodic dataset for few-shot meta-learning.
Supports HAM10000 (primary) and DermaMNIST (legacy fallback).
"""

import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List, Dict, Optional
from PIL import Image
import yaml

from src.data.preprocessing import (
    DermaMNISTPreprocessor,
    ClassBalancedSampler,
    load_ham10000,
    HAM10000_CLASS_NAMES,
    _resolve_config,
)


class EpisodicDermaMNIST(Dataset):
    """
    Episodic dataset for N-way K-shot meta-learning.
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
            self._load_ham10000(config_full_path)
        else:
            self._load_dermamnist(config_full_path, download)

        self.class_to_indices = self._organize_by_class()
        self.class_counts = {c: len(idx) for c, idx in self.class_to_indices.items()}

        if split == "train":
            self.class_sampler = ClassBalancedSampler(
                class_counts=self.class_counts,
                minority_threshold=self.config["dataset"].get("minority_threshold", 200),
                minority_prob=0.4,
            )

        self._print_summary()

    # ------------------------------------------------------------------ #
    # Loaders                                                              #
    # ------------------------------------------------------------------ #

    def _load_ham10000(self, config_full_path: str) -> None:
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

    def _load_dermamnist(self, config_full_path: str, download: bool) -> None:
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

        self.images = dataset.imgs          # (N, 28, 28, 3)
        self.labels = dataset.labels.flatten()
        self.class_names = self.config["dataset"].get("class_names", {})

    # ------------------------------------------------------------------ #
    # Internals                                                            #
    # ------------------------------------------------------------------ #

    def _organize_by_class(self) -> Dict[int, List[int]]:
        class_to_indices: Dict[int, List[int]] = {}
        for idx, label in enumerate(self.labels):
            label = int(label)
            class_to_indices.setdefault(label, []).append(idx)
        return class_to_indices

    def _load_image(self, idx: int) -> torch.Tensor:
        if self._use_paths:
            pil = Image.open(self.image_paths[idx]).convert("RGB")
            return self.preprocessor.preprocess_pil(pil, augment=self.augment)
        else:
            return self.preprocessor.preprocess_image(self.images[idx], augment=self.augment)

    def _print_summary(self) -> None:
        n = len(self.image_paths) if self._use_paths else len(self.images)
        print(f"\nLoaded {'HAM10000' if self._use_paths else 'DermaMNIST'} [{self.split}]:")
        print(f"  Total samples: {n}")
        for cls_id in sorted(self.class_to_indices.keys()):
            name = self.class_names.get(cls_id, str(cls_id))
            print(f"  Class {cls_id} ({name}): {self.class_counts[cls_id]}")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def sample_episode(
        self,
        n_way: int = 5,
        k_shot: int = 5,
        n_query: int = 15,
        return_class_ids: bool = False,
    ) -> Tuple:
        """
        Sample one N-way K-shot episode.

        Returns:
            support_images: (N*K, 3, 224, 224)
            support_labels: (N*K,)      — remapped 0..N-1
            query_images:   (N*Q, 3, 224, 224)
            query_labels:   (N*Q,)      — remapped 0..N-1
        """
        if self.split == "train":
            episode_classes = self.class_sampler.sample_classes(n_way)
        else:
            available = list(self.class_to_indices.keys())
            episode_classes = np.random.choice(available, size=n_way, replace=False).tolist()

        # Remap original class ids → 0..N-1 for loss computation
        label_remap = {orig: new for new, orig in enumerate(episode_classes)}

        support_images, support_labels = [], []
        query_images,   query_labels   = [], []

        for cls_id in episode_classes:
            indices = self.class_to_indices[cls_id]
            n_needed = k_shot + n_query
            replace = len(indices) < n_needed
            sampled = np.random.choice(indices, size=n_needed, replace=replace)

            for i in sampled[:k_shot]:
                support_images.append(self._load_image(i))
                support_labels.append(label_remap[cls_id])

            for i in sampled[k_shot:]:
                query_images.append(self._load_image(i))
                query_labels.append(label_remap[cls_id])

        result = (
            torch.stack(support_images),
            torch.tensor(support_labels, dtype=torch.long),
            torch.stack(query_images),
            torch.tensor(query_labels, dtype=torch.long),
        )
        if return_class_ids:
            return result + (episode_classes,)
        return result

    def __len__(self) -> int:
        if self.split == "train":
            return self.config["few_shot"]["episodes"]
        return self.config["evaluation"]["n_episodes"]

    def __getitem__(self, idx: int) -> Tuple:
        n_way  = self.config["few_shot"]["n_way"]
        k_shot = self.config["few_shot"]["k_shot"]
        n_query = self.config["few_shot"]["n_query"]
        return self.sample_episode(n_way, k_shot, n_query)
