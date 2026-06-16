"""Diagnose the 63% (evaluate.py) vs 54% (my scripts) gap.
Runs image_only / multimodal(forward) / M3-blend over the SAME episodes,
seeded(42) vs unseeded, to see if (a) absolute level is seed-dependent and
(b) the relative ordering (M3 > image_only) is preserved either way."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

N_WAY, K_SHOT, N_QUERY, EP = 5, 5, 15, 200
CKPT = "checkpoints/raresight_nblk4mix.pth"
BLEND = "src/app/assets/blend_params.pt"
DESC = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")

def _n(x): return x / x.norm(dim=-1, keepdim=True)

def run(model, ds, desc, bp, temp, seeded):
    if seeded:
        np.random.seed(42); torch.manual_seed(42)
    beta, lam, gap = bp["beta"], bp["lam"], bp["gap"]
    cupl = bp["text_embs"]
    c_img = c_mm = c_m3 = tot = 0
    with torch.no_grad():
        for _ in range(EP):
            s, _, q, ql, oc = ds.sample_episode(N_WAY, K_SHOT, N_QUERY, return_class_ids=True)
            s, q = s.to(gap.device), q.to(gap.device)
            qe = _n(model.backbone.encode_image(q))
            ip = _n(_n(model.backbone.encode_image(s)).view(N_WAY, K_SHOT, -1).mean(1))
            s_txt = [desc.get(str(c), "dermoscopy image of a skin lesion") for c in oc for _ in range(K_SHOT)]
            mm = _n(model.encode_multimodal(s, s_txt).view(N_WAY, K_SHOT, -1).mean(1))
            txt = _n(torch.stack([cupl[c] for c in oc]) + lam * gap)
            m3 = _n(beta * ip + (1 - beta) * txt)
            ql = ql.numpy()
            for proto, cnt in [(ip, "img"), (mm, "mm"), (m3, "m3")]:
                pred = torch.softmax(-torch.cdist(qe, proto) * temp, 1).argmax(1).cpu().numpy()
                ok = int((pred == ql).sum())
                if cnt == "img": c_img += ok
                elif cnt == "mm": c_mm += ok
                else: c_m3 += ok
            tot += len(ql)
    return 100*c_img/tot, 100*c_mm/tot, 100*c_m3/tot

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    desc = json.load(open(os.path.abspath(DESC)))
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    bp = torch.load(BLEND, map_location=dev)
    temp = m.temperature.item()
    ds = EpisodicDermaMNIST(split="test", augment=False)
    for seeded in [False, True]:
        img, mm, m3 = run(m, ds, desc, bp, temp, seeded)
        print(f"  seeded={seeded!s:<5}  image_only={img:.2f}  multimodal(MLP)={mm:.2f}  M3_blend={m3:.2f}   (Δ M3-img={m3-img:+.2f})")

if __name__ == "__main__":
    main()
