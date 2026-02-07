"""
Baseline 3: Zero-Shot BiomedCLIP
No training - direct classification using medical vision-language model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
import open_clip
from huggingface_hub import hf_hub_download
import json


class ZeroShotBiomedCLIP(nn.Module):
    """
    Zero-shot classification using BiomedCLIP
    Pre-trained on 15M biomedical image-text pairs
    """
    
    def __init__(
        self,
        model_name: str = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        temperature: float = 0.07
    ):
        """
        Args:
            model_name: BiomedCLIP model identifier
            temperature: Temperature for scaling similarity scores
        """
        super(ZeroShotBiomedCLIP, self).__init__()
        
        self.temperature = temperature
        
        print(f"Loading BiomedCLIP model...")
        
        # Load BiomedCLIP from HuggingFace
        try:
            self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
                'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
            )
            self.tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        except Exception as e:
            print(f"Error loading with hf-hub prefix: {e}")
            print("Trying alternative loading method...")
            
            # Alternative: Load pretrained weights
            self.model, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
                'ViT-B-16',
                pretrained='laion2b_s34b_b88k'
            )
            self.tokenizer = open_clip.get_tokenizer('ViT-B-16')
            
            # Try to load BiomedCLIP weights from HuggingFace
            try:
                checkpoint_path = hf_hub_download(
                    "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
                    "open_clip_pytorch_model.bin"
                )
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                self.model.load_state_dict(state_dict)
                print("✓ Loaded BiomedCLIP weights from HuggingFace")
            except Exception as e2:
                print(f"Warning: Could not load BiomedCLIP weights: {e2}")
                print("Using standard CLIP weights instead")
        
        # Freeze all parameters (no training)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        print(f"✓ Model loaded successfully (zero-shot mode)")
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to feature vectors
        
        Args:
            images: Image tensor (B, 3, 224, 224)
        
        Returns:
            Image features (B, feature_dim)
        """
        with torch.no_grad():
            image_features = self.model.encode_image(images)
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)
        return image_features
    
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text descriptions to feature vectors
        
        Args:
            texts: List of text descriptions
        
        Returns:
            Text features (N, feature_dim)
        """
        with torch.no_grad():
            # Tokenize
            text_tokens = self.tokenizer(texts)
            
            # Move to same device as model
            text_tokens = text_tokens.to(next(self.model.parameters()).device)
            
            # Encode
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize features
            text_features = F.normalize(text_features, dim=-1)
        
        return text_features
    
    def predict(
        self,
        images: torch.Tensor,
        text_descriptions: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Zero-shot prediction using image-text similarity
        
        Args:
            images: Image tensor (B, 3, 224, 224)
            text_descriptions: List of N class descriptions
        
        Returns:
            logits: Classification logits (B, N)
            predictions: Predicted class indices (B,)
        """
        # Encode images and texts
        image_features = self.encode_images(images)  # (B, feature_dim)
        text_features = self.encode_texts(text_descriptions)  # (N, feature_dim)
        
        # Compute similarity (cosine similarity since features are normalized)
        similarity = torch.mm(image_features, text_features.t())  # (B, N)
        
        # Scale by temperature
        logits = similarity / self.temperature
        
        # Predict
        predictions = logits.argmax(dim=1)
        
        return logits, predictions
    
    def forward(self, images: torch.Tensor, text_descriptions: List[str]) -> torch.Tensor:
        """
        Forward pass (for compatibility)
        
        Args:
            images: Image tensor (B, 3, 224, 224)
            text_descriptions: List of N class descriptions
        
        Returns:
            logits: Classification logits (B, N)
        """
        logits, _ = self.predict(images, text_descriptions)
        return logits


# Test the model
if __name__ == "__main__":
    print("Testing ZeroShotBiomedCLIP...\n")
    
    # Create model
    model = ZeroShotBiomedCLIP()
    
    # Test with dummy data
    dummy_images = torch.randn(8, 3, 224, 224)
    
    # DermaMNIST class descriptions
    class_descriptions = [
        "A dermoscopy image of actinic keratoses and intraepithelial carcinoma",
        "A dermoscopy image of basal cell carcinoma",
        "A dermoscopy image of benign keratosis-like lesions",
        "A dermoscopy image of dermatofibroma",
        "A dermoscopy image of melanoma",
        "A dermoscopy image of melanocytic nevi",
        "A dermoscopy image of vascular lesions"
    ]
    
    # Zero-shot prediction
    logits, predictions = model.predict(dummy_images, class_descriptions)
    
    print(f"Input shape: {dummy_images.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions.tolist()}")
    
    print("\n✓ Model test successful!")