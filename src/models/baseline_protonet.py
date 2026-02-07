"""
Baseline 2: Standard Prototypical Networks with ResNet-50
Image-only episodic meta-learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple


class StandardProtoNet(nn.Module):
    """
    Standard Prototypical Networks baseline
    ResNet-50 backbone + episodic meta-learning
    """
    
    def __init__(
        self,
        backbone: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        feature_dim: int = 512,
        distance: str = "euclidean"
    ):
        """
        Args:
            backbone: Backbone architecture ("resnet50", "resnet18")
            pretrained: Use ImageNet pre-trained weights
            freeze_backbone: If True, freeze backbone layers
            feature_dim: Dimension of output features
            distance: Distance metric ("euclidean" or "cosine")
        """
        super(StandardProtoNet, self).__init__()
        
        self.feature_dim = feature_dim
        self.distance = distance
        
        # Load backbone
        if backbone == "resnet50":
            self.backbone = models.resnet50(pretrained=pretrained)
            backbone_dim = 2048
        elif backbone == "resnet18":
            self.backbone = models.resnet18(pretrained=pretrained)
            backbone_dim = 512
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        
        # Remove final FC layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"Backbone frozen. Only training projection head.")
        else:
            print(f"Full network trainable.")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images
        
        Args:
            x: Images (B, 3, 224, 224)
        
        Returns:
            Features (B, feature_dim)
        """
        # Backbone features
        features = self.backbone(x)  # (B, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (B, 2048)
        
        # Project
        features = self.projection(features)  # (B, feature_dim)
        
        return features
    
    def _remap_labels(self, labels: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Remap labels to consecutive range [0, n_way-1]
        
        Args:
            labels: Original labels (may be non-consecutive like [0, 2, 4, 5, 6])
        
        Returns:
            remapped_labels: Labels in range [0, n_way-1]
            label_map: Dictionary mapping original → remapped
        """
        unique_labels = torch.unique(labels).tolist()
        label_map = {orig: new for new, orig in enumerate(unique_labels)}
        
        remapped = torch.tensor(
            [label_map[label.item()] for label in labels],
            dtype=labels.dtype,
            device=labels.device
        )
        
        return remapped, label_map
    
    def compute_prototypes(
        self,
        support_features: torch.Tensor,
        support_labels: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """
        Compute class prototypes from support set
        
        Args:
            support_features: Support features (N*K, feature_dim)
            support_labels: Support labels (N*K,) - must be in range [0, n_way-1]
            n_way: Number of classes
        
        Returns:
            Prototypes (n_way, feature_dim)
        """
        prototypes = []
        
        for c in range(n_way):
            # Get features for class c
            class_mask = (support_labels == c)
            class_features = support_features[class_mask]
            
            # Compute mean (prototype)
            prototype = class_features.mean(dim=0)
            prototypes.append(prototype)
        
        prototypes = torch.stack(prototypes)  # (n_way, feature_dim)
        
        return prototypes
    
    def compute_distances(
        self,
        query_features: torch.Tensor,
        prototypes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute distances between queries and prototypes
        
        Args:
            query_features: Query features (N*Q, feature_dim)
            prototypes: Class prototypes (n_way, feature_dim)
        
        Returns:
            Distances (N*Q, n_way)
        """
        if self.distance == "euclidean":
            # Euclidean distance: ||q - p||^2
            # Expand dimensions for broadcasting
            query_features = query_features.unsqueeze(1)  # (N*Q, 1, feature_dim)
            prototypes = prototypes.unsqueeze(0)  # (1, n_way, feature_dim)
            
            # Compute squared distances
            distances = torch.sum((query_features - prototypes) ** 2, dim=2)  # (N*Q, n_way)
            
            # Negative distances (for softmax classification)
            return -distances
        
        elif self.distance == "cosine":
            # Cosine similarity
            # Normalize
            query_features = F.normalize(query_features, dim=1)  # (N*Q, feature_dim)
            prototypes = F.normalize(prototypes, dim=1)  # (n_way, feature_dim)
            
            # Compute cosine similarity
            similarities = torch.mm(query_features, prototypes.t())  # (N*Q, n_way)
            
            return similarities
        
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
    
    def loss(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        query_labels: torch.Tensor,
        n_way: int
    ) -> Tuple[torch.Tensor, float]:
        """
        Compute prototypical loss for one episode
        
        Args:
            support_images: Support images (N*K, 3, 224, 224)
            support_labels: Support labels (N*K,) - original DermaMNIST labels
            query_images: Query images (N*Q, 3, 224, 224)
            query_labels: Query labels (N*Q,) - original DermaMNIST labels
            n_way: Number of classes in episode
        
        Returns:
            loss: Cross-entropy loss
            accuracy: Classification accuracy (%)
        """
        # Remap labels to [0, n_way-1]
        support_labels_remapped, label_map = self._remap_labels(support_labels)
        query_labels_remapped = torch.tensor(
            [label_map[label.item()] for label in query_labels],
            dtype=query_labels.dtype,
            device=query_labels.device
        )
        
        # Extract features
        support_features = self.forward(support_images)  # (N*K, feature_dim)
        query_features = self.forward(query_images)  # (N*Q, feature_dim)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels_remapped, n_way)
        
        # Compute distances/similarities
        logits = self.compute_distances(query_features, prototypes)  # (N*Q, n_way)
        
        # Cross-entropy loss
        loss = F.cross_entropy(logits, query_labels_remapped)
        
        # Accuracy
        _, predicted = logits.max(1)
        accuracy = predicted.eq(query_labels_remapped).float().mean().item() * 100
        
        return loss, accuracy
    
    def predict(
        self,
        support_images: torch.Tensor,
        support_labels: torch.Tensor,
        query_images: torch.Tensor,
        n_way: int
    ) -> torch.Tensor:
        """
        Predict labels for query images
        
        Args:
            support_images: Support images (N*K, 3, 224, 224)
            support_labels: Support labels (N*K,) - original labels
            query_images: Query images (N*Q, 3, 224, 224)
            n_way: Number of classes
        
        Returns:
            Predicted labels (N*Q,) - remapped to [0, n_way-1]
        """
        with torch.no_grad():
            # Remap support labels
            support_labels_remapped, _ = self._remap_labels(support_labels)
            
            # Extract features
            support_features = self.forward(support_images)
            query_features = self.forward(query_images)
            
            # Compute prototypes
            prototypes = self.compute_prototypes(support_features, support_labels_remapped, n_way)
            
            # Compute distances
            logits = self.compute_distances(query_features, prototypes)
            
            # Predict
            _, predicted = logits.max(1)
        
        return predicted


# Test the model
if __name__ == "__main__":
    print("Testing StandardProtoNet...\n")
    
    # Create model
    model = StandardProtoNet(
        backbone="resnet50",
        pretrained=True,
        freeze_backbone=False,
        feature_dim=512,
        distance="euclidean"
    )
    
    # Test episode with NON-CONSECUTIVE labels (like real DermaMNIST)
    n_way = 5
    k_shot = 5
    n_query = 15
    
    # Simulate real episode labels: [0, 2, 3, 5, 6] (not consecutive)
    episode_classes = [0, 2, 3, 5, 6]
    
    support_images = torch.randn(n_way * k_shot, 3, 224, 224)
    support_labels = torch.tensor([cls for cls in episode_classes for _ in range(k_shot)])
    query_images = torch.randn(n_way * n_query, 3, 224, 224)
    query_labels = torch.tensor([cls for cls in episode_classes for _ in range(n_query)])
    
    print(f"Episode classes (original): {episode_classes}")
    print(f"Support labels: {support_labels[:10].tolist()}...")
    print(f"Query labels: {query_labels[:10].tolist()}...")
    
    # Forward pass
    model.train()
    loss, accuracy = model.loss(
        support_images, support_labels,
        query_images, query_labels,
        n_way
    )
    
    print(f"\nLoss: {loss.item():.4f}")
    print(f"Accuracy: {accuracy:.2f}%")
    
    # Test prediction
    model.eval()
    predictions = model.predict(support_images, support_labels, query_images, n_way)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions (remapped): {predictions[:10].tolist()}...")
    
    print("\nModel test successful!")