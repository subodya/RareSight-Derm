import torch
import torch.nn as nn
import open_clip
from transformers import AutoTokenizer


class RareSight(nn.Module):
    def __init__(self, device='cuda'):
        super(RareSight, self).__init__()
        self.device = device

        print("Loading BiomedCLIP backbone...")
        self.model_name = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        self.backbone, _, self.preprocess = open_clip.create_model_and_transforms(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )

        # Freeze entire backbone by default.
        # train_advanced.py will selectively unfreeze the last N ViT blocks.
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Residual fusion network (trainable)
        self.fusion_net = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
        )

        # Learnable gating scalar
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Learnable temperature — scales prototype distances before softmax.
        # Initialized to 10.0 (typical for cosine-space metric learning).
        self.temperature = nn.Parameter(torch.tensor(10.0))

        self.to(device)

    def encode_multimodal(self, images, text_list):
        inputs = self.tokenizer(
            text_list, padding='max_length', truncation=True,
            max_length=256, return_tensors='pt'
        )
        text_tokens = inputs['input_ids'].to(self.device)

        # Image encoding — no no_grad() here so that gradients flow through
        # any unfrozen ViT blocks into the fusion network and the loss.
        # Fully-frozen layers already have requires_grad=False and incur no
        # extra memory for gradient storage.
        img_emb = self.backbone.encode_image(images)
        img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

        # Text encoder is always kept frozen — no_grad is safe here.
        with torch.no_grad():
            text_emb = self.backbone.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        # Gated residual fusion
        combined = torch.cat((img_emb, text_emb), dim=1)
        residual = self.fusion_net(combined)
        fused = img_emb + self.alpha * residual
        fused = fused / fused.norm(dim=-1, keepdim=True)

        return fused

    def forward(self, support_images, support_texts, query_images, n_way, k_shot):
        # Support path — gradients flow (fusion net + unfrozen ViT blocks trained here)
        support_emb = self.encode_multimodal(support_images, support_texts)
        prototypes = support_emb.view(n_way, k_shot, -1).mean(dim=1)
        prototypes = prototypes / prototypes.norm(dim=-1, keepdim=True)

        # Query path — query images bypass the fusion net; no grad needed here
        with torch.no_grad():
            query_emb = self.backbone.encode_image(query_images)
            query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

        dists = torch.cdist(query_emb, prototypes)
        return -dists * self.temperature  # temperature-scaled negative distance → logits
