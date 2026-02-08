import torch
import torch.nn as nn
import open_clip
from transformers import AutoTokenizer

class RareSight(nn.Module):
    def __init__(self, device='cuda'):
        super(RareSight, self).__init__()
        self.device = device
        
        # 1. Load Pre-trained BiomedCLIP (Frozen Backbone)
        print("Loading BiomedCLIP backbone...")
        self.model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained('microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        
        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Residual Fusion Network
        self.fusion_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512) 
        )
        
        # Learnable gating parameter
        self.alpha = nn.Parameter(torch.tensor(0.1))
        
        self.to(device)

    def encode_multimodal(self, images, text_list):
        # 1. Tokenize
        inputs = self.tokenizer(
            text_list, padding='max_length', truncation=True, 
            max_length=256, return_tensors='pt'
        )
        text_tokens = inputs['input_ids'].to(self.device)
        
        with torch.no_grad():
            # 2. Get Raw Features (Frozen)
            img_emb = self.backbone.encode_image(images)
            # FIX: Use out-of-place division
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            
            text_emb = self.backbone.encode_text(text_tokens)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            
        # 3. Calculate Residual Update
        combined = torch.cat((img_emb, text_emb), dim=1) 
        residual = self.fusion_net(combined)             
        
        # 4. Apply Residual Connection
        fused = img_emb + (self.alpha * residual)
        
        # 5. Re-normalize (FIX: Out-of-place)
        fused = fused / fused.norm(dim=-1, keepdim=True)
        
        return fused

    def forward(self, support_images, support_texts, query_images, n_way, k_shot):
        # 1. Get Support Prototypes (Fused)
        support_emb = self.encode_multimodal(support_images, support_texts)
        
        # Average K-shot samples to get class prototypes
        prototypes = support_emb.view(n_way, k_shot, -1).mean(dim=1)
        
        # FIX: Out-of-place division for normalization
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)
        
        # 2. Get Query Embeddings (Raw Image)
        with torch.no_grad():
            query_emb = self.backbone.encode_image(query_images)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)
            
        # 3. Euclidean Distance
        dists = torch.cdist(query_emb, prototypes)
        
        return -dists # Negative distance = Logits