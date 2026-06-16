"""
Evaluate a set of (staged) deployment artifacts across the RareSight outputs that
matter: per-band classification accuracy, abstention (Mahalanobis OOD), referral
rate (entropy gate), and calibration — on HAM-test per band and real DermaMNIST-test.

Mirrors the deployed DERMOSCOPY path in inference.predict (band routing -> prototype
logits -> maha OOD tau -> temperature calibration -> entropy referral). Image-only
(no metadata/note) so the encoder effect is isolated and base-vs-new is apples-to-apples.

Reads encoder from RS_CKPT and artifacts from RS_OUT_DIR (both env).
Run:  RS_CKPT=checkpoints/raresight_nblk4mix.pth RS_OUT_DIR=src/app/assets_staging_new \
        python src/app/tools/eval_deployed.py
"""
import sys, os, json
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.models.raresight_net import RareSight
import src.app.tools.build_serving_artifacts as bsa

ROOT   = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
CKPT   = os.environ.get("RS_CKPT", os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth"))
ASSET  = os.environ.get("RS_OUT_DIR", os.path.join(ROOT, "src", "app", "assets"))
HAM    = os.path.join(ROOT, "data", "ham10000")
NPZ    = os.path.join(ROOT, "data", "raw", "dermamnist.npz")
BANDS  = [28, 56, 112, 224, 450]
N      = 7
SEED   = 42
EVAL_CAP = 900
dev = "cuda" if torch.cuda.is_available() else "cpu"


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    print(f"device={dev}  ckpt={os.path.basename(CKPT)}  assets={os.path.basename(ASSET)}")
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()

    protos = torch.load(os.path.join(ASSET, "disease_prototypes_multi.pt"), map_location="cpu")
    serving = torch.load(os.path.join(ASSET, "serving_params_multi.pt"),
                         map_location="cpu", weights_only=False)

    @torch.no_grad()
    def enc_pils(pils, bs=64):
        out = []
        for i in range(0, len(pils), bs):
            t = torch.stack([m.preprocess(p) for p in pils[i:i + bs]]).to(dev)
            out.append(_norm(m.backbone.encode_image(t)).cpu())
        return torch.cat(out).numpy()

    def evaluate(Q, y, band):
        sv = serving["per_band"][band]
        P = torch.stack([_norm(protos[band][c]) for c in range(N)]).numpy()
        z = -(np.linalg.norm(Q[:, None, :] - P[None], axis=2)) * temp     # cdist logits
        pred = z.argmax(1)
        acc = 100.0 * (pred == y).mean()
        # Mahalanobis OOD -> abstain
        means = np.asarray(sv["maha_means"]); inv = np.asarray(sv["maha_inv_cov"])
        diff = Q[:, None, :] - means[None]
        score = -np.einsum("nci,ij,ncj->nc", diff, inv, diff).min(1)
        is_unknown = score < sv["ood_tau"]
        # referral = OOD OR high entropy (image-only regime, all 7 classes available)
        T = sv["calib_T_img"]
        p = bsa.softmax_np(z / T)
        ent = -(p * np.log(p + 1e-12)).sum(1)
        refer = is_unknown | (ent > 0.75 * np.log(N))
        ece = bsa.ece(p, y)
        committed = ~refer
        acc_committed = 100.0 * (pred[committed] == y[committed]).mean() if committed.any() else float("nan")
        per_class = {}
        for c in range(N):
            mk = (y == c)
            per_class[c] = round(100.0 * (pred[mk] == c).mean(), 1) if mk.any() else None
        return {"acc": round(acc, 2), "abstain%": round(100 * is_unknown.mean(), 1),
                "refer%": round(100 * refer.mean(), 1),
                "acc_committed": round(acc_committed, 2), "ece": round(ece, 4),
                "per_class_recall": per_class, "support": {c: int((y == c).sum()) for c in range(N)},
                "n": len(y)}

    sp = bsa.load_split_meta(HAM, seed=SEED)
    te = sp["test"].reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    if len(te) > EVAL_CAP:
        te = te.iloc[rng.choice(te.index.to_numpy(), EVAL_CAP, replace=False)].reset_index(drop=True)
    yt = te["label"].values

    print(f"\n{'band':>6} {'acc':>7} {'acc_commit':>11} {'abstain%':>9} {'refer%':>8} {'ece':>7} {'n':>5}")
    results = {"ckpt": os.path.basename(CKPT), "assets": os.path.basename(ASSET), "ham": {}, }
    for band in BANDS:
        pils = [Image.open(p).convert("RGB").resize((band, band), Image.BILINEAR)
                if band < min(Image.open(p).size) else Image.open(p).convert("RGB")
                for p in te["path"]]
        Q = enc_pils(pils)
        r = evaluate(Q, yt, band)
        results["ham"][band] = r
        print(f"{band:>6} {r['acc']:>7} {r['acc_committed']:>11} {r['abstain%']:>9} "
              f"{r['refer%']:>8} {r['ece']:>7} {r['n']:>5}")

    # real DermaMNIST-test @28 (citable)
    d = np.load(NPZ); dte_x, dte_y = d["test_images"], d["test_labels"].ravel()
    Qd = enc_pils([Image.fromarray(x) for x in dte_x])
    rd = evaluate(Qd, dte_y, 28)
    results["dermamnist_test28"] = rd
    print(f"{'DM-28':>6} {rd['acc']:>7} {rd['acc_committed']:>11} {rd['abstain%']:>9} "
          f"{rd['refer%']:>8} {rd['ece']:>7} {rd['n']:>5}")

    # per-class recall (top-1) for the two most telling rows
    CN = {0: "akiec", 1: "bcc", 2: "bkl", 3: "df", 4: "mel", 5: "nv", 6: "vasc"}
    print(f"\nper-class recall %  ({'  '.join(CN[c] for c in range(N))})")
    for tag, r in [("HAM@224", results["ham"][224]), ("DM-28", results["dermamnist_test28"])]:
        pc = r["per_class_recall"]; sup = r["support"]
        cells = "  ".join(f"{(pc[c] if pc[c] is not None else 0):5.1f}" for c in range(N))
        print(f"  {tag:>8}: {cells}")
        print(f"  {'(n)':>8}: " + "  ".join(f"{sup[c]:5d}" for c in range(N)))

    out = os.path.join(ROOT, "thesis", f"eval_deployed_{os.path.basename(ASSET)}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
