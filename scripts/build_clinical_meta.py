"""
FIX #1 — wire metadata fusion into the deployed CLINICAL path.

Builds class-conditional log P(age|c), P(sex|c), P(site|c) tables on PAD-UFES-20 PAD-TRAIN and
adds them to src/app/assets/clinical_serving_params.pt with the SAME keys the dermoscopy path
uses (meta_logtab / meta_sex_index / meta_site_index / meta_alpha / calib_T_meta). Once present,
inference.py:388 fires metadata fusion on the clinical branch automatically — no code change.

Protocol: tables fit on PAD-TRAIN; meta_alpha tuned on a patient-grouped val carve (macro-F1);
calib_T_meta fit on the fused train logits. Reuses the exact 15-entry HAM site index + sex index
so the frontend dropdown values map to real columns. df/vasc rows stay uniform (masked on the
clinical path anyway).

Backs up the artifact first. Run:  python scripts/build_clinical_meta.py
"""
import sys, os, json, shutil, numpy as np, torch
from datetime import date
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight
from src.data.pad_ufes import load_pad_ufes

CKPT = os.path.join(ROOT, "checkpoints", "raresight_nblk4mix.pth")
CLIN_PROTO = os.path.join(ROOT, "src/app/assets/clinical_prototypes.pt")
CLIN_SERVING = os.path.join(ROOT, "src/app/assets/clinical_serving_params.pt")

N = 7
SMOOTH = 1.0
ALPHA_GRID = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
SEX_IDX = {"male": 0, "female": 1, "unknown": 2}
SITE_IDX = {"abdomen": 0, "acral": 1, "back": 2, "chest": 3, "ear": 4, "face": 5, "foot": 6,
            "genital": 7, "hand": 8, "lower extremity": 9, "neck": 10, "scalp": 11, "trunk": 12,
            "unknown": 13, "upper extremity": 14}


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


def age_bin(a):
    try:
        a = float(a)
    except (TypeError, ValueError):
        return None
    return None if a != a else int(min(max(a, 0), 89) // 10)


def norm_sex(s):
    s = str(s).strip().lower()
    return s if s in ("male", "female") else "unknown"


def fit_tables(df):
    age = np.full((N, 9), SMOOTH); sex = np.full((N, 3), SMOOTH); site = np.full((N, len(SITE_IDX)), SMOOTH)
    for _, r in df.iterrows():
        c = int(r["label"]); ab = age_bin(r["age"])
        if ab is not None:
            age[c, ab] += 1
        sex[c, SEX_IDX[norm_sex(r["sex"])]] += 1
        site[c, SITE_IDX.get(str(r["site"]).strip().lower(), SITE_IDX["unknown"])] += 1
    return {"age": np.log(age / age.sum(1, keepdims=True)).tolist(),
            "sex": np.log(sex / sex.sum(1, keepdims=True)).tolist(),
            "site": np.log(site / site.sum(1, keepdims=True)).tolist()}


def meta_ll(df, lt):
    A, S, T = np.array(lt["age"]), np.array(lt["sex"]), np.array(lt["site"])
    out = np.zeros((len(df), N))
    for i, (_, r) in enumerate(df.iterrows()):
        ab = age_bin(r["age"])
        if ab is not None:
            out[i] += A[:, ab]
        out[i] += S[:, SEX_IDX[norm_sex(r["sex"])]]
        out[i] += T[:, SITE_IDX.get(str(r["site"]).strip().lower(), SITE_IDX["unknown"])]
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    cs = torch.load(CLIN_SERVING, map_location="cpu", weights_only=False)
    temp = float(cs["temp_metric"]); classes = [int(c) for c in cs["classes"]]
    protos = torch.load(CLIN_PROTO, map_location="cpu", weights_only=False)
    P = np.stack([_norm(protos[c]).numpy() for c in classes]).astype(np.float32)

    pad = load_pad_ufes(verbose=False)
    train_all, test = pad["train"], pad["test"]
    gss = GroupShuffleSplit(1, test_size=0.25, random_state=42)
    tr_i, va_i = next(gss.split(train_all, train_all["label"], groups=train_all["patient_id"]))
    tr, va = train_all.iloc[tr_i].reset_index(drop=True), train_all.iloc[va_i].reset_index(drop=True)

    def proto_logits(X):
        z = -np.linalg.norm(X[:, None, :] - P[None], axis=2) * temp
        full = np.full((len(X), N), -1e9); full[:, classes] = z
        return full

    print("Encoding PAD train/val/test...")
    Xva = encode(m, dev, va["path"].tolist()); yva = va["label"].values.astype(int)
    Xtr_all = encode(m, dev, train_all["path"].tolist()); ytr_all = train_all["label"].values.astype(int)
    Xte = encode(m, dev, test["path"].tolist()); yte = test["label"].values.astype(int)

    # tune alpha on val (tables fit on tr only)
    lt_tr = fit_tables(tr)
    zva, mva = proto_logits(Xva), meta_ll(va, lt_tr)
    mf1 = lambda y, p: f1_score(y, p, average="macro", labels=classes, zero_division=0)
    best_a, best = 0.0, -1
    for a in ALPHA_GRID:
        f = mf1(yva, (zva + a * mva).argmax(1))
        if f > best:
            best, best_a = f, a
    print(f"meta_alpha* (val macro-F1) = {best_a}")

    # final tables on ALL train; calib_T_meta on fused train logits
    lt_final = fit_tables(train_all)
    ztr, mtr = proto_logits(Xtr_all), meta_ll(train_all, lt_final)
    fused_tr = ztr + best_a * mtr

    def fit_T(logits, y):
        bT, bn = 1.0, 1e9
        for T in np.concatenate([np.linspace(0.05, 1.0, 96), np.linspace(1.05, 10, 90)]):
            e = np.exp(logits / T - (logits / T).max(1, keepdims=True)); p = e / e.sum(1, keepdims=True)
            nll = -np.mean(np.log(p[np.arange(len(y)), y] + 1e-12))
            if nll < bn:
                bn, bT = nll, float(T)
        return bT
    T_meta = fit_T(fused_tr, ytr_all)
    print(f"calib_T_meta = {T_meta:.3f}")

    # held-out check
    zte, mte = proto_logits(Xte), meta_ll(test, lt_final)
    acc_img = (zte.argmax(1) == yte).mean() * 100
    f1_img = mf1(yte, zte.argmax(1)) * 100
    pred_meta = (zte + best_a * mte).argmax(1)
    acc_meta = (pred_meta == yte).mean() * 100
    f1_meta = mf1(yte, pred_meta) * 100
    print(f"\nPAD-TEST (held-out) check:")
    print(f"  image-only : acc {acc_img:.2f}%  macroF1 {f1_img:.2f}")
    print(f"  + metadata : acc {acc_meta:.2f}%  macroF1 {f1_meta:.2f}  "
          f"(d_acc {acc_meta-acc_img:+.2f}, d_F1 {f1_meta-f1_img:+.2f})")

    # back up + patch the deployed artifact
    bak = CLIN_SERVING.replace(".pt", f"_backup_{date.today().strftime('%Y%m%d')}_premeta.pt")
    if not os.path.exists(bak):
        shutil.copy2(CLIN_SERVING, bak)
        print(f"\nbacked up -> {os.path.relpath(bak, ROOT)}")
    cs["meta_logtab"] = lt_final
    cs["meta_sex_index"] = SEX_IDX
    cs["meta_site_index"] = SITE_IDX
    cs["meta_alpha"] = float(best_a)
    cs["calib_T_meta"] = round(float(T_meta), 4)
    cs["calib_T_img"] = float(cs.get("calib_T", 1.0))
    cs["meta_source"] = "PAD-UFES-20 train; built scripts/build_clinical_meta.py"
    torch.save(cs, CLIN_SERVING)
    print(f"patched  -> {os.path.relpath(CLIN_SERVING, ROOT)}  (added meta_logtab, meta_alpha={best_a}, calib_T_meta)")
    json.dump({"meta_alpha": best_a, "calib_T_meta": round(T_meta, 4),
               "padtest_image_only": {"acc": round(acc_img, 2), "macroF1": round(f1_img, 2)},
               "padtest_with_metadata": {"acc": round(acc_meta, 2), "macroF1": round(f1_meta, 2)}},
              open(os.path.join(ROOT, "checkpoints", "clinical_meta_wired_results.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
