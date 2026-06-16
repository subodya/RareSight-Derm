"""
CoCoOp-inspired *support-conditional* prompt learning for RareSight.

CoOp (coop_prompt.py) learns ONE static context vector shared across all classes and
all episodes. Its weakness (Zhou et al., CVPR'22): the static context overfits the
trained classes and transfers poorly to novel ones. CoCoOp fixes this by making the
context *input-conditional* — a lightweight Meta-Net produces a per-input shift added
to every context token.

ADAPTATION FOR THIS ARCHITECTURE (state honestly in the thesis):
  Vanilla CoCoOp conditions on each *query image*. Here text features are blended into
  a class *prototype* (M3: proto = beta*img + (1-beta)*(text+lam*gap)), so a per-query
  prompt would mean rebuilding every prototype per query (a prototype stops being a
  fixed class representation, and it is N_query x slower). Instead we condition on each
  class's **support prototype** ip_c (mean of the K support embeddings). The learned
  text for class c thus adapts to the actual support images of *this* episode —
  including unseen/novel classes — which is exactly the generalization CoCoOp targets.
  Call this "CoCoOp-inspired / support-conditional prompting", not vanilla CoCoOp.

Trainable surface = ctx (M x 768) + Meta-Net (512 -> 512/16 -> 768). Backbone frozen;
blend beta/lam/gap fixed at M3. Meta-Net final layer init ~0 so training STARTS as CoOp
(shift ~ 0) and only departs if it earns it. Reuses all cached embeddings from CoOp.
"""

import torch
import torch.nn as nn
from src.models.coop_prompt import CoOpPromptLearner


class CoCoOpPromptLearner(CoOpPromptLearner):
    def __init__(self, model, class_names: dict, n_ctx: int = 4,
                 ctx_init: str = "a dermoscopy image of a", device: str = "cuda",
                 vis_dim: int = 512, bottleneck: int = 32):
        super().__init__(model, class_names, n_ctx=n_ctx, ctx_init=ctx_init, device=device)
        hidden = self.word_emb.weight.shape[1]          # 768 (ctx token dim)

        # Meta-Net: image-prototype (512) -> token-space shift (768), shared by all M tokens.
        fc2 = nn.Linear(bottleneck, hidden)
        nn.init.zeros_(fc2.weight); nn.init.zeros_(fc2.bias)     # start as CoOp (shift=0)
        self.meta_net = nn.Sequential(
            nn.Linear(vis_dim, bottleneck),
            nn.ReLU(inplace=True),
            fc2,
        ).to(device)
        self._last_shift_mag = 0.0                      # diagnostic: mean |shift|

    def trainable_params(self):
        return [self.ctx] + list(self.meta_net.parameters())

    def _build_inputs_cond(self, cls_ids, ctx_per_class):
        """ctx_per_class: (n, M, 768) — a distinct context block per class."""
        seqs, masks = [], []
        for j, cid in enumerate(cls_ids):
            name = self.class_name_emb[cid]
            seq = torch.cat([self.cls_emb, ctx_per_class[j], name, self.sep_emb], 0)
            L = seq.shape[0]
            mask = torch.ones(L, device=self.device)
            if L < self.max_len:
                pad = self.pad_emb.expand(self.max_len - L, -1)
                seq = torch.cat([seq, pad], 0)
                mask = torch.cat([mask, torch.zeros(self.max_len - L, device=self.device)], 0)
            seqs.append(seq); masks.append(mask)
        return torch.stack(seqs), torch.stack(masks)

    def text_features(self, cls_ids: list, img_protos: torch.Tensor = None,
                      normalize: bool = True):
        """img_protos: (n, 512) normalized support prototypes, one per class in cls_ids.
        If None, falls back to static CoOp behaviour (shift = 0)."""
        n = len(cls_ids)
        if img_protos is None:
            ctx_per_class = self.ctx.unsqueeze(0).expand(n, -1, -1)
            self._last_shift_mag = 0.0
        else:
            shift = self.meta_net(img_protos)                    # (n, 768)
            self._last_shift_mag = float(shift.detach().abs().mean())
            ctx_per_class = self.ctx.unsqueeze(0) + shift.unsqueeze(1)   # (n, M, 768)
        emb, mask = self._build_inputs_cond(cls_ids, ctx_per_class)
        feat = self._encode(emb, mask)
        if normalize:
            feat = feat / feat.norm(dim=-1, keepdim=True)
        return feat
