"""
Preprocessing utilities for RareSight-Derm.
Supports both DermaMNIST (numpy arrays) and HAM10000 (PIL images from file paths).
"""

import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, List, Optional, Dict
import yaml
from pathlib import Path

# HAM10000 dx value → integer class (mirrors DermaMNIST label ordering).
# Includes both the short abbreviations (Kaggle CSV) and the full names
# used by the marmal88/skin_cancer HuggingFace download.
HAM10000_LABEL_MAP: Dict[str, int] = {
    # Short abbreviations (original Kaggle metadata)
    "akiec": 0,
    "bcc":   1,
    "bkl":   2,
    "df":    3,
    "mel":   4,
    "nv":    5,
    "vasc":  6,
    # Full names (marmal88/skin_cancer HuggingFace download)
    "actinic_keratoses":          0,
    "basal_cell_carcinoma":       1,
    "benign_keratosis-like_lesions": 2,
    "dermatofibroma":             3,
    "melanoma":                   4,
    "melanocytic_Nevi":           5,
    "vascular_lesions":           6,
}

HAM10000_CLASS_NAMES: Dict[int, str] = {
    0: "Actinic keratoses and intraepithelial carcinoma",
    1: "Basal cell carcinoma",
    2: "Benign keratosis-like lesions",
    3: "Dermatofibroma",
    4: "Melanoma",
    5: "Melanocytic nevi",
    6: "Vascular lesions",
}


# ---------------------------------------------------------------------------
# Config helper
# ---------------------------------------------------------------------------

def _resolve_config(config_path: str) -> str:
    if os.path.isabs(config_path) or os.path.exists(config_path):
        return config_path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    full = os.path.join(project_root, config_path)
    if not os.path.exists(full):
        raise FileNotFoundError(f"Config not found: {config_path}")
    return full


# ---------------------------------------------------------------------------
# Image preprocessor (works for both numpy arrays and PIL images)
# ---------------------------------------------------------------------------

class DermaMNISTPreprocessor:
    """
    Preprocesses skin lesion images for BiomedCLIP (224×224, ImageNet norm).
    Accepts either numpy uint8 arrays (DermaMNIST) or PIL.Image objects (HAM10000).
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        config_full_path = _resolve_config(config_path)
        with open(config_full_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.target_size = self.config["dataset"]["target_size"]
        self.mean = self.config["dataset"]["mean"]
        self.std  = self.config["dataset"]["std"]
        self.transform = self._build_transform()
        self.aug_transform = self._build_aug_transform()

    def _build_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize(
                (self.target_size, self.target_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def _build_aug_transform(self) -> transforms.Compose:
        return transforms.Compose([
            transforms.RandomResizedCrop(
                self.target_size, scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    # ------------------------------------------------------------------ #
    # Public API — accepts PIL.Image or numpy array                        #
    # ------------------------------------------------------------------ #

    def preprocess_pil(self, image: Image.Image, augment: bool = False) -> torch.Tensor:
        """Preprocess a PIL.Image (HAM10000 path)."""
        image = image.convert("RGB")
        tfm = self.aug_transform if augment else self.transform
        return tfm(image)

    def preprocess_image(self, image: np.ndarray, augment: bool = False) -> torch.Tensor:
        """Preprocess a numpy uint8 array (DermaMNIST format)."""
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 1:
            image = np.repeat(image, 3, axis=-1)

        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)

        pil = Image.fromarray(image)
        return self.preprocess_pil(pil, augment=augment)

    def preprocess_batch(self, images: np.ndarray, augment: bool = False) -> torch.Tensor:
        return torch.stack([self.preprocess_image(img, augment) for img in images])

    def denormalize(self, tensor: torch.Tensor) -> np.ndarray:
        t = tensor.clone()
        for c, m, s in zip(t, self.mean, self.std):
            c.mul_(s).add_(m)
        return torch.clamp(t, 0, 1).permute(1, 2, 0).cpu().numpy()


# ---------------------------------------------------------------------------
# HAM10000 loader
# ---------------------------------------------------------------------------

def load_ham10000(
    data_root: str,
    split: str,
    val_size: float = 0.1,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[List[str], np.ndarray]:
    """
    Load HAM10000 dataset from disk.

    Expects this layout under `data_root`:
        HAM10000_metadata.csv
        <any subdirectories or root>/<image_id>.jpg

    Args:
        data_root:  Path to HAM10000 root directory.
        split:      "train", "val", or "test".
        val_size:   Fraction for validation split.
        test_size:  Fraction for test split.
        seed:       Random seed for reproducibility.

    Returns:
        image_paths:  List[str] — absolute paths to JPEG files.
        labels:       np.ndarray[int] — integer class labels (0–6).
    """
    meta_csv = os.path.join(data_root, "HAM10000_metadata.csv")
    if not os.path.exists(meta_csv):
        raise FileNotFoundError(
            f"HAM10000_metadata.csv not found in {data_root}.\n"
            "Download HAM10000 from https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection\n"
            f"and extract it to:  {data_root}"
        )

    meta = pd.read_csv(meta_csv)

    # Build image_id → path map by scanning all subdirectories
    id_to_path: Dict[str, str] = {}
    for jpg in glob.glob(os.path.join(data_root, "**", "*.jpg"), recursive=True):
        img_id = os.path.splitext(os.path.basename(jpg))[0]
        id_to_path[img_id] = jpg
    # Also check root for flat layouts
    for jpg in glob.glob(os.path.join(data_root, "*.jpg")):
        img_id = os.path.splitext(os.path.basename(jpg))[0]
        if img_id not in id_to_path:
            id_to_path[img_id] = jpg

    # Keep only rows with a found image
    meta = meta[meta["image_id"].isin(id_to_path)].copy()
    if len(meta) == 0:
        raise FileNotFoundError(
            f"No .jpg images found under {data_root}. "
            "Make sure the HAM10000 image folders (HAM10000_images_part1, etc.) are present."
        )

    meta["label"] = meta["dx"].map(HAM10000_LABEL_MAP)
    missing = meta["label"].isna()
    if missing.any():
        unknown = meta.loc[missing, "dx"].unique().tolist()
        raise ValueError(f"Unknown dx labels in metadata: {unknown}")

    meta["label"] = meta["label"].astype(int)
    meta["path"] = meta["image_id"].map(id_to_path)

    all_paths = meta["path"].values
    all_labels = meta["label"].values

    # Stratified split: train / val / test
    # First split off test
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(sss_test.split(all_paths, all_labels))

    # Then split val from train
    val_fraction = val_size / (1.0 - test_size)
    sss_val = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
    train_idx, val_idx = next(sss_val.split(all_paths[train_val_idx], all_labels[train_val_idx]))
    train_idx = train_val_idx[train_idx]
    val_idx   = train_val_idx[val_idx]

    idx_map = {"train": train_idx, "val": val_idx, "test": test_idx}
    if split not in idx_map:
        raise ValueError(f"split must be 'train', 'val', or 'test', got '{split}'")

    chosen = idx_map[split]
    return list(all_paths[chosen]), all_labels[chosen]


# ---------------------------------------------------------------------------
# Class-balanced sampler (unchanged public API)
# ---------------------------------------------------------------------------

class ClassBalancedSampler:
    """
    Samples N-way episodes with increased probability of minority classes.
    """

    def __init__(
        self,
        class_counts: Dict[int, int],
        minority_threshold: int = 200,
        minority_prob: float = 0.4,
    ):
        self.class_counts = class_counts
        self.minority_threshold = minority_threshold
        self.minority_prob = minority_prob

        self.minority_classes = [c for c, n in class_counts.items() if n < minority_threshold]
        self.majority_classes = [c for c, n in class_counts.items() if n >= minority_threshold]

        print(f"  Minority classes (<{minority_threshold}): {self.minority_classes}")
        print(f"  Majority classes (≥{minority_threshold}): {self.majority_classes}")

    def sample_classes(self, n_way: int) -> List[int]:
        include_minority = (
            len(self.minority_classes) > 0
            and np.random.rand() < self.minority_prob
        )
        all_classes = self.minority_classes + self.majority_classes

        if include_minority and n_way > 1:
            n_min = min(np.random.randint(1, 3), len(self.minority_classes), n_way)
            minority_pick = np.random.choice(self.minority_classes, size=n_min, replace=False)
            majority_pick = np.random.choice(self.majority_classes, size=n_way - n_min, replace=False)
            return np.concatenate([minority_pick, majority_pick]).tolist()

        return np.random.choice(all_classes, size=n_way, replace=False).tolist()
