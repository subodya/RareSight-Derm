"""
STEP 1 (the prize) — HAM->PAD CROSS-DATASET GENERALIZATION EVAL.

Tests the DEPLOYED dermoscopy model (frozen BiomedCLIP + M3 disease_prototypes.pt + the
serving OOD/calibration) on PAD-UFES-20 CLINICAL photos it has never seen — the off-modality
distribution real users actually upload. NO PAD TRAINING: this is pure zero-PAD generalization,
the gold-standard answer to "does this work on real-world cases?".

Evaluated on the reserved PAD-TEST split (patient-grouped; see src/data/pad_ufes.py). Reports
THREE things (per the design): (a) accuracy on the 5 shared classes under realistic full-7way
argmax, (b) the full 7-way confusion matrix, (c) the OOD ABSTENTION RATE — expected to be high
(clinical photos are off-modality), which is itself the headline finding and the motivation for
the clinical-adaptation steps 2-4.

Run: conda run -n raresight python src/training/eval_cross_dataset_pad.py
"""
import sys, os, json, numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes, HAM_NAMES, SHARED_HAM_CLASSES

N = 7
CKPT = "checkpoints/raresight_nblk4mix.pth"
PROTO = "src/app/assets/disease_prototypes.pt"
SERVING = "src/app/assets/serving_params.pt"
OUT = "checkpoints/cross_dataset_pad_results.json"


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def encode(model, dev, paths, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            b = torch.cat([model.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                           for p in paths[i:i+bs]]).to(dev)
            out.append(_norm(model.backbone.encode_image(b)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def recalls(pred, y, classes):
    return {c: (float((pred[y == c] == c).mean()) if (y == c).any() else None) for c in classes}


def confusion(pred, y, true_classes):
    M = np.zeros((N, N), int)
    for t, p in zip(y, pred):
        M[t, p] += 1
    print("\nConfusion (rows=true PAD class, cols=predicted HAM class, row %):")
    print("                 " + " ".join(f"{HAM_NAMES[c][:6]:>6}" for c in range(N)))
    for t in true_classes:
        row = M[t] / max(M[t].sum(), 1) * 100
        print(f"  {HAM_NAMES[t]:<15} " + " ".join(f"{row[p]:6.0f}" for p in range(N)))
    return M.tolist()


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()

    pad = load_pad_ufes()
    test = pad["test"]
    paths = test["path"].tolist()
    y = test["label"].values.astype(int)

    P = torch.stack([_norm(torch.load(PROTO, map_location="cpu")[c]) for c in range(N)]).numpy()
    serving = torch.load(SERVING, map_location="cpu", weights_only=False) if os.path.exists(SERVING) else None
    temp = float(serving["temp_metric"]) if serving else float(m.temperature.item())

    print(f"\nEncoding {len(paths)} PAD-TEST clinical images...")
    X = encode(m, dev, paths)

    # prototype logits + predictions (realistic full-7way argmax)
    z = -np.linalg.norm(X[:, None, :] - P[None], axis=2) * temp     # (n, 7)
    pred_full = z.argmax(1)
    # oracle restricted to the 5 shared classes (upper bound)
    mask = np.full(N, -1e9); mask[SHARED_HAM_CLASSES] = 0.0
    pred_shared = (z + mask).argmax(1)

    # OOD abstention (Mahalanobis on the embedding — the deployed safety net)
    abstain = None
    if serving and serving.get("ood_method") == "maha":
        means = np.array(serving["maha_means"]); inv = np.array(serving["maha_inv_cov"])
        diff = X[:, None, :] - means[None]                          # (n,7,512)
        d = np.einsum("nci,ij,ncj->nc", diff, inv, diff)
        ood = -d.min(1)
        tau = float(serving["ood_tau"])
        abstain = ood < tau

    # ── report ──
    acc_full = float((pred_full == y).mean() * 100)
    acc_shared = float((pred_shared == y).mean() * 100)
    rec_full = recalls(pred_full, y, SHARED_HAM_CLASSES)
    print("\n" + "=" * 64)
    print("HAM->PAD CROSS-DATASET (zero PAD training, PAD-TEST, chance=20%)")
    print("=" * 64)
    print(f"Accuracy (realistic full-7way argmax) : {acc_full:5.2f}%")
    print(f"Accuracy (oracle restricted 5-way)    : {acc_shared:5.2f}%")
    print("Per-class recall (full-7way):")
    for c in SHARED_HAM_CLASSES:
        r = rec_full[c]
        print(f"  {HAM_NAMES[c]:22s}: {('%.1f%%' % (r*100)) if r is not None else 'n/a':>7}")
    res = {"n_test": len(y), "accuracy_full7way": round(acc_full, 2),
           "accuracy_oracle5way": round(acc_shared, 2),
           "per_class_recall_full7way": {HAM_NAMES[c]: (round(rec_full[c]*100, 1) if rec_full[c] is not None else None)
                                         for c in SHARED_HAM_CLASSES},
           "temp": round(temp, 4)}

    if abstain is not None:
        ab_rate = float(abstain.mean() * 100)
        kept = ~abstain
        acc_kept = float((pred_full[kept] == y[kept]).mean() * 100) if kept.any() else None
        print(f"\nOOD ABSTENTION RATE (safety net flags as 'unknown'): {ab_rate:5.2f}%")
        print(f"  → the deployed dermoscopy guard correctly refuses {ab_rate:.0f}% of off-modality clinical photos")
        print(f"  accuracy among NON-abstained: {acc_kept if acc_kept is None else round(acc_kept,2)}%")
        res["ood_abstention_rate"] = round(ab_rate, 2)
        res["accuracy_non_abstained"] = round(acc_kept, 2) if acc_kept is not None else None

    res["confusion_full7way"] = confusion(pred_full, y, SHARED_HAM_CLASSES)
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved → {OUT}")
    print("\nINTERPRETATION: a HIGH abstention rate is the finding, not a bug — it quantifies the\n"
          "domain shift you observed on the webp/clinical samples and motivates the clinical path (steps 2-4).")


if __name__ == "__main__":
    main()
