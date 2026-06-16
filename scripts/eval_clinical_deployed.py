"""
Load-and-eval of the CURRENTLY DEPLOYED clinical path on PAD-UFES-20 PAD-TEST.

Tests the EXACT artifacts the live backend serves for clinical photos:
  encoder  = checkpoints/raresight_nblk4mix.pth   (deployed, fine-tuned nblk4mix)
  protos   = src/app/assets/clinical_prototypes.pt
  serving  = src/app/assets/clinical_serving_params.pt   (maha OOD + temp)

No rebuild, no training, no deployed file is modified. Encoder and prototypes are a
MATCHED pair (both nblk4mix, Jun-10) — unlike eval_cross_dataset_pad.py, whose single-res
disease_prototypes.pt is a stale frozen-encoder artifact. Scoring replicates the clinical
path in build_clinical_path.py: euclidean prototype logits * temp, argmax over the 5 shared
clinical classes; Mahalanobis abstention < tau.

Run:  python scripts/eval_clinical_deployed.py
"""
import sys, os, json, numpy as np, torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes, HAM_NAMES

CKPT = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")
PROTO = os.path.join(ROOT, "src/app/assets/clinical_prototypes.pt")
SERVING = os.path.join(ROOT, "src/app/assets/clinical_serving_params.pt")
OUT = os.path.join(ROOT, "checkpoints", "clinical_deployed_padtest_results.json")


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def encode(model, dev, paths, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            b = torch.cat([model.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                           for p in paths[i:i + bs]]).to(dev)
            out.append(_norm(model.backbone.encode_image(b)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}\nEncoder: {os.path.basename(CKPT)} (deployed clinical artifacts)")
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()

    proto = torch.load(PROTO, map_location="cpu", weights_only=False)
    sp = torch.load(SERVING, map_location="cpu", weights_only=False)
    classes = [int(c) for c in sp["classes"]]
    temp = float(sp["temp_metric"])
    tau = float(sp["ood_tau"])
    means = np.asarray(sp["maha_means"], dtype=np.float32)
    inv = np.asarray(sp["maha_inv_cov"], dtype=np.float32)
    P = np.stack([_norm(proto[c]).numpy() for c in classes]).astype(np.float32)

    pad = load_pad_ufes()
    test = pad["test"]
    paths = test["path"].tolist()
    y = test["label"].values.astype(int)
    # restrict reporting to the 5 shared clinical classes present in PAD-test
    keep = np.isin(y, classes)
    paths = [p for p, k in zip(paths, keep) if k]
    y = y[keep]

    print(f"Encoding {len(paths)} PAD-TEST clinical images (classes {classes})...")
    X = encode(m, dev, paths)

    # euclidean prototype logits * temp, argmax over clinical classes (deployed scoring)
    z = -np.linalg.norm(X[:, None, :] - P[None], axis=2) * temp     # (n, 5)
    pred = np.array(classes)[z.argmax(1)]

    # Mahalanobis abstention (deployed safety net)
    diff = X[:, None, :] - means[None]
    d = np.einsum("nci,ij,ncj->nc", diff, inv, diff)
    ood = -d.min(1)
    abstain = ood < tau

    acc = float((pred == y).mean() * 100)
    ab_rate = float(abstain.mean() * 100)
    kept = ~abstain
    acc_kept = float((pred[kept] == y[kept]).mean() * 100) if kept.any() else None
    rec = {HAM_NAMES[c]: (round(float((pred[y == c] == c).mean()) * 100, 1), int((y == c).sum()))
           for c in classes if (y == c).any()}

    print("\n" + "=" * 60)
    print("DEPLOYED CLINICAL PATH on PAD-TEST (5-way, chance=20%)")
    print("=" * 60)
    print(f"  n_test              : {len(y)}")
    print(f"  accuracy (all)      : {acc:5.2f}%")
    print(f"  OOD abstention rate : {ab_rate:5.2f}%")
    print(f"  accuracy (non-abst) : {acc_kept if acc_kept is None else round(acc_kept, 2)}%")
    print("  per-class recall (n):")
    for c in classes:
        if HAM_NAMES[c] in rec:
            r, n = rec[HAM_NAMES[c]]
            print(f"    {HAM_NAMES[c]:22s}: {r:5.1f}%  (n={n})")

    res = {"encoder": "raresight_nblk4mix.pth", "artifacts": "deployed clinical (src/app/assets)",
           "n_test": int(len(y)), "clinical_classes": classes,
           "accuracy": round(acc, 2), "ood_abstention_rate": round(ab_rate, 2),
           "accuracy_non_abstained": round(acc_kept, 2) if acc_kept is not None else None,
           "per_class_recall_pct": {k: v[0] for k, v in rec.items()},
           "per_class_n": {k: v[1] for k, v in rec.items()},
           "temp_metric": round(temp, 4), "ood_tau": round(tau, 4)}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
