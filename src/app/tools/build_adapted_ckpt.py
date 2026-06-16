"""
Inject the Phase-2 fine-tuned visual encoder (nblk4 + real-DM mix winner) into a
RareSight checkpoint so the serving builders can regenerate artifacts from it.

The fine-tune only touched backbone.visual (last 4 ViT blocks + final norms + CLIP
projection); the text tower, fusion_net, temperature etc. are taken from the deployed
raresight_finetuned.pth unchanged. Output is a NEW checkpoint — the deployed one is
left untouched.

Run:  python src/app/tools/build_adapted_ckpt.py
"""
import sys, os, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight

ROOT     = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
SRC_CKPT = os.path.join(ROOT, "checkpoints", "raresight_finetuned.pth")
FT_CKPT  = os.path.join(ROOT, "checkpoints", "finetune_multires", "nblk4_seed42_mix0.5_best.pth")
OUT_CKPT = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")


def main():
    dev = "cpu"
    print(f"Loading RareSight base: {SRC_CKPT}")
    rs = RareSight(device=dev)
    rs.load_state_dict(torch.load(SRC_CKPT, map_location=dev), strict=False)

    print(f"Loading fine-tuned visual: {FT_CKPT}")
    ft = torch.load(FT_CKPT, map_location=dev)
    vis_state = ft["visual"]

    missing, unexpected = rs.backbone.visual.load_state_dict(vis_state, strict=True)
    # strict=True raises on mismatch; if we got here keys matched exactly.
    print(f"  injected {len(vis_state)} visual tensors (keys matched strict=True)")

    # sanity: confirm a last-block weight actually differs from the base encoder
    base = RareSight(device=dev)
    base.load_state_dict(torch.load(SRC_CKPT, map_location=dev), strict=False)
    k = "trunk.blocks.11.mlp.fc2.weight"
    d = (rs.backbone.visual.state_dict()[k] - base.backbone.visual.state_dict()[k]).abs().max().item()
    print(f"  max|adapted - base| on {k} = {d:.4f}  (expect > 0)")
    k0 = "trunk.blocks.0.mlp.fc2.weight"
    d0 = (rs.backbone.visual.state_dict()[k0] - base.backbone.visual.state_dict()[k0]).abs().max().item()
    print(f"  max|adapted - base| on {k0} = {d0:.4f}  (expect ~0, block 0 frozen)")

    torch.save(rs.state_dict(), OUT_CKPT)
    print(f"\nWROTE adapted checkpoint -> {OUT_CKPT}")


if __name__ == "__main__":
    main()
