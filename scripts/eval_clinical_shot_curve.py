"""
CLINICAL FEW-SHOT SHOT-CURVE on PAD-UFES-20 (smartphone photos).

Question: can you stand up the clinical classifier from only a HANDFUL of smartphone
prototypes, instead of the ~290 images/class the deployed clinical path uses?

For each K in {1,5,10,20} we build clinical prototypes from K *distinct patients* per class
(one image per patient — the honest few-shot unit; the deployed build uses ALL images,
multiple per patient), using the SAME M3 recipe as build_clinical_path.py (beta=0.75,
lam=1.0, CuPL text + modality gap RECOMPUTED from each sample's support). We then score
5-way argmax on the reserved PAD-TEST and average over several seeds (few-shot is high
variance). Effective K is capped at available patients per class (melanoma is thin).

Anchored against two references already measured live:
    chance (5-way)                  = 20.0%
    dermoscopy prototypes, 5-way    = 55.6%   (zero clinical data — the floor to beat)
    full-data clinical path, 5-way  = 66.0%   (deployed; ALL ~290 img/class — the ceiling)
The headline is the crossover K: the smallest K whose clinical prototypes beat 55.6%.

NOTE: this isolates PROTOTYPE quality (fixed temp, argmax). The Mahalanobis OOD gate needs
a 512-dim covariance (>>512 samples) and genuinely cannot be estimated at low K — the
safety net still needs data even where the classifier does not.

Run:  conda run -n raresight python scripts/eval_clinical_shot_curve.py
Env:  SC_KS (default "1,5,10,20"), SC_SEEDS (default 8)
"""
import sys, os, json, numpy as np, torch
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes, HAM_NAMES, SHARED_HAM_CLASSES

CKPT = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")
BLEND = os.path.join(ROOT, "src/app/assets/blend_params.pt")
CACHE = os.path.join(ROOT, "checkpoints", "_pad_emb_cache.npz")
OUT = os.path.join(ROOT, "checkpoints", "clinical_shot_curve_results.json")
BETA, LAM = 0.75, 1.0
KS = [int(k) for k in os.environ.get("SC_KS", "1,5,10,20").split(",")]
SEEDS = int(os.environ.get("SC_SEEDS", "8"))
DERMO_5WAY, FULL_DATA = 55.6, 66.0   # live references (see module docstring)


def _norm_t(x):
    return x / x.norm(dim=-1, keepdim=True)


def _norm_np(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def encode(model, dev, paths, bs=16):
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), bs):
            b = torch.cat([model.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                           for p in paths[i:i + bs]]).to(dev)
            out.append(_norm_t(model.backbone.encode_image(b)).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}")
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()
    temp = float(m.temperature.item())

    pad = load_pad_ufes(verbose=False)
    train, test = pad["train"], pad["test"]
    classes = [c for c in SHARED_HAM_CLASSES if (train["label"].values == c).any()]

    # ── encode once, cache ──
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        Xtr, Xte = z["Xtr"], z["Xte"]
        ytr, yte = z["ytr"], z["yte"]
        pid = z["pid"]
        print(f"Loaded cached embeddings ({len(Xtr)} train / {len(Xte)} test).")
    else:
        print(f"Encoding PAD-TRAIN ({len(train)}) + PAD-TEST ({len(test)})...")
        Xtr = encode(m, dev, train["path"].tolist()); ytr = train["label"].values.astype(int)
        Xte = encode(m, dev, test["path"].tolist());  yte = test["label"].values.astype(int)
        pid = train["patient_id"].astype(str).values
        np.savez(CACHE, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, pid=pid)
        print(f"Cached -> {os.path.relpath(CACHE, ROOT)}")

    # CuPL text per class (shared, K-independent)
    blend = torch.load(BLEND, map_location="cpu", weights_only=False)
    text = {c: _norm_t(blend["text_embs"][c]).numpy().astype(np.float32) for c in classes}

    # patients available per class (the few-shot sampling unit)
    cls_patients = {c: sorted(set(pid[ytr == c])) for c in classes}
    avail = {c: len(cls_patients[c]) for c in classes}
    print("Distinct PAD-TRAIN patients per class (caps effective K):")
    for c in classes:
        print(f"  {HAM_NAMES[c]:22s}: {avail[c]}")
    # index rows by (class, patient) so we can pick one image per sampled patient
    rows_of = {(c, p): np.where((ytr == c) & (pid == p))[0] for c in classes for p in cls_patients[c]}

    def build_and_eval(k, seed):
        rng = np.random.RandomState(seed)
        img_protos, realized = {}, {}
        for c in classes:
            pats = cls_patients[c]
            keff = min(k, len(pats))
            realized[c] = keff
            chosen = rng.choice(len(pats), size=keff, replace=False)
            idxs = [int(rng.choice(rows_of[(c, pats[j])])) for j in chosen]  # one img/patient
            img_protos[c] = _norm_np(Xtr[idxs].mean(0))
        # modality gap recomputed from THIS sample's support (honest at every K)
        img_mean = _norm_np(np.mean([img_protos[c] for c in classes], axis=0))
        txt_mean = _norm_np(np.mean([text[c] for c in classes], axis=0))
        gap = _norm_np(img_mean - txt_mean)
        P_img, P_m3 = [], []
        for c in classes:
            P_img.append(img_protos[c])
            txt_shift = _norm_np(text[c] + LAM * gap)
            P_m3.append(_norm_np(BETA * img_protos[c] + (1 - BETA) * txt_shift))
        P_img = np.stack(P_img); P_m3 = np.stack(P_m3)
        cls_arr = np.array(classes)
        keep = np.isin(yte, classes)
        Xk, yk = Xte[keep], yte[keep]

        def acc(P):
            z = -np.linalg.norm(Xk[:, None, :] - P[None], axis=2) * temp
            return float((cls_arr[z.argmax(1)] == yk).mean() * 100)
        return acc(P_img), acc(P_m3), realized

    print(f"\nClinical few-shot shot-curve (5-way, {SEEDS} seeds/K, chance=20%)")
    print(f"  references: dermoscopy-protos={DERMO_5WAY}%  full-data clinical={FULL_DATA}%\n")
    print(f"{'K':>4} {'realized':>20}  {'image-only':>18}  {'M3-blend':>18}  {'beats 55.6%?':>12}")
    curve = {}
    for k in KS:
        ai, am, real = [], [], None
        for s in range(SEEDS):
            a_img, a_m3, real = build_and_eval(k, 1000 + s)
            ai.append(a_img); am.append(a_m3)
        ai, am = np.array(ai), np.array(am)
        rstr = ",".join(str(real[c]) for c in classes)
        beats = "YES" if am.mean() > DERMO_5WAY else "no"
        print(f"{k:>4} {rstr:>20}  {ai.mean():6.2f} +/- {ai.std():4.2f}   "
              f"{am.mean():6.2f} +/- {am.std():4.2f}   {beats:>12}")
        curve[str(k)] = {
            "realized_k_per_class": {HAM_NAMES[c]: real[c] for c in classes},
            "image_only_mean": round(float(ai.mean()), 2), "image_only_std": round(float(ai.std()), 2),
            "m3_mean": round(float(am.mean()), 2), "m3_std": round(float(am.std()), 2),
            "m3_beats_dermoscopy": bool(am.mean() > DERMO_5WAY),
        }

    res = {"seeds": SEEDS, "classes": classes, "temp": round(temp, 4),
           "ref_chance_5way": 20.0, "ref_dermoscopy_5way": DERMO_5WAY,
           "ref_full_data_clinical_5way": FULL_DATA,
           "patients_per_class": {HAM_NAMES[c]: avail[c] for c in classes},
           "sampling_unit": "one image per distinct patient", "curve": curve}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {os.path.relpath(OUT, ROOT)}")
    print("NOTE: this measures PROTOTYPE quality only. The Mahalanobis OOD gate needs a "
          "512-dim covariance (>>512 samples) and cannot be built at low K.")


if __name__ == "__main__":
    main()
