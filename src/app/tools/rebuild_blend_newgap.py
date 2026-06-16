"""
Recompute the modality GAP on the fine-tuned encoder + re-tune global BETA, then write a
staged blend_params. The visual-only fine-tune moved the image manifold, so the deployed gap
(= mean_img - mean_text, fit on the FROZEN encoder) is stale; the text-blend term is therefore
mis-aligned for the new encoder. This realigns it.

gap recipe mirrors precompute.py:197 (gap = norm(mean_img - mean_text)); beta is re-tuned on VAL
(HAM full-res 224 + DermaMNIST-28) so a single global beta fits the new encoder.

Writes: <RS_OUT_DIR>/blend_params.pt = {gap(new), beta(retuned), lam, text_embs(CoOp, unchanged)}.
Feed it to the builders via RS_BLEND. Non-destructive (staging only).

Run:  RS_CKPT=checkpoints/raresight_nblk4mix.pth RS_OUT_DIR=src/app/assets_staging_newgap \
        python src/app/tools/rebuild_blend_newgap.py
"""
import sys, os, random
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
from src.data.preprocessing import load_ham10000
import src.app.tools.build_serving_artifacts as bsa

ROOT  = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CKPT  = os.environ.get("RS_CKPT", os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth"))
ASSET = os.environ.get("RS_OUT_DIR", os.path.join(ROOT, "src", "app", "assets_staging_newgap"))
LIVE_BLEND = os.path.join(ROOT, "src", "app", "assets", "blend_params.pt")
HAM   = os.path.join(ROOT, "data", "ham10000")
NPZ   = os.path.join(ROOT, "data", "raw", "dermamnist.npz")
K_SHOT, SEED, N = 20, 42, 7
BETA_GRID = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
VAL_CAP = 500
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def _select_diverse(paths, labels, c, k, seed):
    cls = [p for p, l in zip(paths, labels) if int(l) == c]
    random.Random(seed + c).shuffle(cls)
    return cls[:k]


def main():
    os.makedirs(ASSET, exist_ok=True)
    print(f"device={dev}  ckpt={os.path.basename(CKPT)}")
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    bp = torch.load(LIVE_BLEND, map_location=dev, weights_only=False)
    LAM, TXT, OLD_GAP, OLD_BETA = bp["lam"], bp["text_embs"], bp["gap"].to(dev), bp["beta"]

    @torch.no_grad()
    def enc_pils(pils, bs=64):
        out = []
        for i in range(0, len(pils), bs):
            t = torch.stack([m.preprocess(p) for p in pils[i:i + bs]]).to(dev)
            out.append(_norm(m.backbone.encode_image(t)))
        return torch.cat(out)

    def open_full(p): return Image.open(p).convert("RGB")
    def to28(im): return im.resize((28, 28), Image.BILINEAR)

    tr_paths, tr_labels = load_ham10000(HAM, split="train", val_size=0.10, test_size=0.10, seed=SEED)

    # ---- new gap = norm(mean_img - mean_text) on the new encoder ----
    img_full, all_e = {}, []
    for c in range(N):
        sup = _select_diverse(tr_paths, tr_labels, c, K_SHOT, SEED)
        e = enc_pils([open_full(p) for p in sup])
        img_full[c] = _norm(e.mean(0))
        all_e.append(e)
    img_mean = _norm(torch.cat(all_e).mean(0))
    txt_mean = _norm(torch.stack([_norm(TXT[c].to(dev)) for c in range(N)]).mean(0))
    gap_new = _norm(img_mean - txt_mean)
    cos = float((_norm(OLD_GAP) @ gap_new).item())
    print(f"old vs new gap cosine = {cos:.4f}  (1.0 = unchanged; lower = more re-alignment)")

    # ---- val sets for beta re-tune ----
    sp = bsa.load_split_meta(HAM, seed=SEED)
    va = sp["val"].reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    if len(va) > VAL_CAP:
        va = va.iloc[rng.choice(va.index.to_numpy(), VAL_CAP, replace=False)].reset_index(drop=True)
    Qv_ham = enc_pils([open_full(p) for p in va["path"]]); yv_ham = va["label"].values
    d = np.load(NPZ)
    dva_x, dva_y = d["val_images"], d["val_labels"].ravel()
    dtr_x, dtr_y = d["train_images"], d["train_labels"].ravel()
    Qv_dm = enc_pils([Image.fromarray(x) for x in dva_x])
    dm_proto28 = {}
    for c in range(N):
        idx = np.where(dtr_y == c)[0]
        if len(idx):
            pick = rng.choice(idx, min(K_SHOT, len(idx)), replace=False)
            dm_proto28[c] = _norm(enc_pils([to28(Image.fromarray(dtr_x[i])) for i in pick]).mean(0))
        else:
            dm_proto28[c] = _norm(TXT[c].to(dev))

    def acc(Q, protos_dict, y):
        P = torch.stack([protos_dict[c] for c in range(N)])
        pred = (Q @ P.t()).argmax(1).cpu().numpy()
        return 100.0 * (pred == y).mean()

    print(f"\n{'beta':>6} {'HAM@224':>8} {'DM@28':>7} {'mean':>7}")
    best_beta, best_m = OLD_BETA, -1
    for beta in BETA_GRID:
        ts = {c: _norm(TXT[c].to(dev) + LAM * gap_new) for c in range(N)}
        ph = {c: _norm(beta * img_full[c] + (1 - beta) * ts[c]) for c in range(N)}
        pd = {c: _norm(beta * dm_proto28[c] + (1 - beta) * ts[c]) for c in range(N)}
        ah, ad = acc(Qv_ham, ph, yv_ham), acc(Qv_dm, pd, dva_y)
        mean = 0.5 * (ah + ad)
        flag = ""
        if mean > best_m:
            best_m, best_beta, flag = mean, beta, "  <-"
        print(f"{beta:>6.2f} {ah:>8.2f} {ad:>7.2f} {mean:>7.2f}{flag}")
    print(f"\nbest global beta (val) = {best_beta}  (deployed was {OLD_BETA})")

    torch.save({"gap": gap_new.cpu(), "beta": best_beta, "lam": LAM, "text_embs": TXT},
               os.path.join(ASSET, "blend_params.pt"))
    print(f"WROTE staged blend -> {os.path.join(ASSET, 'blend_params.pt')}")


if __name__ == "__main__":
    main()
