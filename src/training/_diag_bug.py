"""Find the 54 vs 63 bug. Run model.forward() AND a manual multimodal recon over
the SAME episodes at several seeds; also replicate the eval_ablation ordering
(image_only computed BEFORE vs AFTER multimodal) to test for state coupling."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

N_WAY, K_SHOT, N_QUERY, EP = 5, 5, 15, 150
CKPT = "checkpoints/raresight_nblk4mix.pth"
DESC = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")
def _n(x): return x / x.norm(dim=-1, keepdim=True)

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    desc = json.load(open(os.path.abspath(DESC)))
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()
    ds = EpisodicDermaMNIST(split="test", augment=False)

    for seed in [None, 42, 123, 7]:
        if seed is not None:
            np.random.seed(seed); torch.manual_seed(seed)
        c_fwd = c_recon = c_img = tot = 0
        with torch.no_grad():
            for _ in range(EP):
                s, _, q, ql, oc = ds.sample_episode(N_WAY, K_SHOT, N_QUERY, return_class_ids=True)
                s, q, ql = s.to(dev), q.to(dev), ql.to(dev)
                s_txt = [desc.get(str(c), "dermoscopy image of a skin lesion") for c in oc for _ in range(K_SHOT)]
                # forward()
                fwd = m(s, s_txt, q, N_WAY, K_SHOT).argmax(1)
                # manual recon
                qe = _n(m.backbone.encode_image(q))
                mm = _n(m.encode_multimodal(s, s_txt).view(N_WAY, K_SHOT, -1).mean(1))
                rec = torch.softmax(-torch.cdist(qe, mm) * temp, 1).argmax(1)
                # image-only
                ip = _n(_n(m.backbone.encode_image(s)).view(N_WAY, K_SHOT, -1).mean(1))
                img = torch.softmax(-torch.cdist(qe, ip) * temp, 1).argmax(1)
                c_fwd += int((fwd == ql).sum()); c_recon += int((rec == ql).sum())
                c_img += int((img == ql).sum()); tot += ql.numel()
        print(f"  seed={str(seed):<5} forward={100*c_fwd/tot:.2f}  recon_mm={100*c_recon/tot:.2f}  image_only={100*c_img/tot:.2f}")

if __name__ == "__main__":
    main()
