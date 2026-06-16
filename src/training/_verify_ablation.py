"""
Sanity check: does the ablation's reconstructed 'multimodal' path equal the
deployed model.forward()? And how does image-only compare on the SAME episodes?
Run over a fixed-seed set of episodes; all three must be apples-to-apples.

    conda run -n raresight python src/training/_verify_ablation.py
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

N_WAY, K_SHOT, N_QUERY, N_EPISODES, SEED = 5, 5, 15, 80, 123
CKPT = "checkpoints/raresight_nblk4mix.pth"
DESC = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")

def _norm(x): return x / x.norm(dim=-1, keepdim=True)

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    desc = json.load(open(os.path.abspath(DESC)))
    ds = EpisodicDermaMNIST(split="test", augment=False)
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()
    temp = m.temperature.item()

    np.random.seed(SEED); torch.manual_seed(SEED)
    c_fwd = c_mm = c_img = tot = 0
    with torch.no_grad():
        for _ in range(N_EPISODES):
            s_img, _, q_img, q_lbl, orig = ds.sample_episode(N_WAY, K_SHOT, N_QUERY, return_class_ids=True)
            s_img, q_img = s_img.to(dev), q_img.to(dev)
            q_lbl = q_lbl.cpu().numpy()
            s_txt = [desc.get(str(c), "dermoscopy image of a skin lesion") for c in orig for _ in range(K_SHOT)]

            # (1) deployed forward()
            logits = m(s_img, s_txt, q_img, N_WAY, K_SHOT)
            p_fwd = logits.argmax(1).cpu().numpy()

            # (2) reconstructed multimodal
            q_emb = _norm(m.backbone.encode_image(q_img))
            proto_mm = _norm(m.encode_multimodal(s_img, s_txt).view(N_WAY, K_SHOT, -1).mean(1))
            p_mm = torch.softmax(-torch.cdist(q_emb, proto_mm) * temp, 1).argmax(1).cpu().numpy()

            # (3) image-only
            proto_img = _norm(_norm(m.backbone.encode_image(s_img)).view(N_WAY, K_SHOT, -1).mean(1))
            p_img = torch.softmax(-torch.cdist(q_emb, proto_img) * temp, 1).argmax(1).cpu().numpy()

            c_fwd += (p_fwd == q_lbl).sum(); c_mm += (p_mm == q_lbl).sum()
            c_img += (p_img == q_lbl).sum(); tot += len(q_lbl)

    print(f"\nOver {N_EPISODES} episodes (seed={SEED}):")
    print(f"  forward()      : {100*c_fwd/tot:.2f}%")
    print(f"  recon multimodal: {100*c_mm/tot:.2f}%   (should ~= forward)")
    print(f"  image_only     : {100*c_img/tot:.2f}%")
    print(f"  match forward==recon: {c_fwd == c_mm}")

if __name__ == "__main__":
    main()
