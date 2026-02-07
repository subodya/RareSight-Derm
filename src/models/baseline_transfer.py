"""
Baseline 1: Transfer Learning with ResNet-50
Standard supervised learning approach
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional


class TransferLearningBaseline(nn.Module):
    """
    ResNet-50 transfer learning baseline
    Pre-trained on ImageNet, fine-tuned on DermaMNIST
    """
    
    def __init__(
        self,
        num_classes: int = 7,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.5
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Use ImageNet pre-trained weights
            freeze_backbone: If True, only train classifier head
            dropout: Dropout rate before final layer
        """
        super(TransferLearningBaseline, self).__init__()
        
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained ResNet-50
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Get feature dimension
        in_features = self.backbone.fc.in_features  # 2048 for ResNet-50
        
        # Replace final fully-connected layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            self._freeze_backbone()
            print(f"Backbone frozen. Only training classifier head.")
        else:
            print(f"Full network trainable ({self.count_parameters():,} parameters)")
    
    def _freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for name, param in self.backbone.named_parameters():
            if 'fc' not in name:  # Don't freeze final layer
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            Logits (B, num_classes)
        """
        return self.backbone(x)
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before final layer
        Useful for visualization (t-SNE)
        
        Args:
            x: Input images (B, 3, 224, 224)
        
        Returns:
            Features (B, 2048)
        """
        # Forward through all layers except final fc
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x


# Test the model
if __name__ == "__main__":
    # Create model
    model = TransferLearningBaseline(
        num_classes=7,
        pretrained=True,
        freeze_backbone=False
    )
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Trainable parameters: {model.count_parameters():,}")
    
    # Test feature extraction
    features = model.get_features(dummy_input)
    print(f"Feature shape: {features.shape}")
    
    print("\nModel test successful!")