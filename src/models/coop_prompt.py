"""
CoOp / BiomedCoOp-style learnable text prompts for BiomedCLIP (PubMedBERT text encoder).

Replaces hand-written/CuPL text with M learnable context vectors prepended to each
class name, in BiomedCLIP's word-embedding space:

    prompt_c = [CLS] ctx_1 .. ctx_M  {class-name tokens}  [SEP]

The per-class text feature is  proj( BertModel(inputs_embeds).last_hidden_state[:,0] ),
i.e. exactly BiomedCLIP's text path (ClsLastHiddenStatePooler @ pos 0 + proj), but with
the leading context tokens made learnable.

Only `ctx` is trainable; the BiomedCLIP backbone stays frozen.

IMPORTANT: call `verify_forward()` once before training — it confirms the manual
inputs_embeds path reproduces the stock encode_text() to <1e-4. If it fails, the
training signal is meaningless (see advisor note).
"""

import torch
import torch.nn as nn

CLS_ID, SEP_ID, PAD_ID = 2, 3, 0


class CoOpPromptLearner(nn.Module):
    def __init__(self, model, class_names: dict, n_ctx: int = 4,
                 ctx_init: str = "a dermoscopy image of a", device: str = "cuda"):
        super().__init__()
        self.device = device
        self.txt = model.backbone.text            # HFTextEncoder
        self.tr = self.txt.transformer            # BertModel
        self.proj = self.txt.proj                 # 768 -> 512
        self.word_emb = self.tr.embeddings.word_embeddings  # (30522, 768)
        self.tokenizer = model.tokenizer
        self.class_ids = sorted(class_names.keys())

        hidden = self.word_emb.weight.shape[1]    # 768

        # ── Initialise context vectors ──────────────────────────────────────
        if ctx_init:
            init_ids = self.tokenizer(ctx_init, add_special_tokens=False,
                                      return_tensors="pt")["input_ids"][0][:n_ctx]
            with torch.no_grad():
                ctx0 = self.word_emb(init_ids.to(device)).clone()   # (<=n_ctx, 768)
            if ctx0.shape[0] < n_ctx:                                # pad with small noise
                pad = 0.02 * torch.randn(n_ctx - ctx0.shape[0], hidden, device=device)
                ctx0 = torch.cat([ctx0, pad], 0)
        else:
            ctx0 = 0.02 * torch.randn(n_ctx, hidden, device=device)
        self.ctx = nn.Parameter(ctx0)             # (M, 768)  TRAINABLE
        self.n_ctx = n_ctx

        # ── Cache fixed (non-learnable) token embeddings per class ──────────
        #   prefix = [CLS] ;  suffix_c = {class tokens} [SEP]
        self.cls_emb = self._emb(torch.tensor([CLS_ID], device=device))      # (1,768)
        self.sep_emb = self._emb(torch.tensor([SEP_ID], device=device))      # (1,768)
        self.pad_emb = self._emb(torch.tensor([PAD_ID], device=device))      # (1,768)
        self.class_name_emb = {}                  # cls_id -> (L_c, 768)
        for cid in self.class_ids:
            name = class_names[cid]
            ids = self.tokenizer(name, add_special_tokens=False,
                                 return_tensors="pt")["input_ids"][0].to(device)
            self.class_name_emb[cid] = self._emb(ids)                         # (L_c,768)

        # max sequence length = CLS + ctx + longest class name + SEP
        self.max_len = 1 + n_ctx + max(e.shape[0] for e in self.class_name_emb.values()) + 1

    def _emb(self, ids):
        with torch.no_grad():
            return self.word_emb(ids).clone()

    def _build_inputs(self, cls_ids: list):
        """Return (inputs_embeds, attn_mask) for the given class ids, padded to max_len."""
        seqs, masks = [], []
        for cid in cls_ids:
            name = self.class_name_emb[cid]
            seq = torch.cat([self.cls_emb, self.ctx, name, self.sep_emb], 0)  # (L,768)
            L = seq.shape[0]
            mask = torch.ones(L, device=self.device)
            if L < self.max_len:
                pad = self.pad_emb.expand(self.max_len - L, -1)
                seq = torch.cat([seq, pad], 0)
                mask = torch.cat([mask, torch.zeros(self.max_len - L, device=self.device)], 0)
            seqs.append(seq)
            masks.append(mask)
        return torch.stack(seqs), torch.stack(masks)                          # (n,max_len,768),(n,max_len)

    def _encode(self, inputs_embeds, attn_mask):
        out = self.tr(inputs_embeds=inputs_embeds, attention_mask=attn_mask.long())
        pooled = out.last_hidden_state[:, 0]      # ClsLastHiddenStatePooler @ pos 0
        return self.proj(pooled)                  # (n, 512)  un-normalised

    def text_features(self, cls_ids: list, normalize: bool = True):
        emb, mask = self._build_inputs(cls_ids)
        feat = self._encode(emb, mask)
        if normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat

    # ── Self-test: manual inputs_embeds path must match stock encode_text ──
    @torch.no_grad()
    def verify_forward(self, test_str="a dermoscopy image of melanoma"):
        ids = self.tokenizer(test_str, padding="max_length", truncation=True,
                             max_length=256, return_tensors="pt")["input_ids"].to(self.device)
        ref = self.txt(ids)                                              # stock path
        emb = self.word_emb(ids)
        mask = (ids != PAD_ID).float()
        manual = self._encode(emb, mask)
        diff = (ref - manual).abs().max().item()
        return diff
