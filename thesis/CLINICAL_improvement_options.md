# Clinical-image accuracy — options to improve PAD-UFES-20 path (all classes)

**Status:** research / ranked options. Nothing implemented yet — pick what to pursue.
**Date:** 2026-06-12

---

## 1. Where the clinical path actually stands (the honest baseline)

The clinical path is the **smartphone/clinical-photo** branch (`clinical_prototypes.pt` +
`clinical_serving_params.pt`), routed to by the modality probe. It is built from **PAD-UFES-20**
(~2,298 clinical photos, patient-grouped split). Measured on held-out **PAD-TEST (n=638)**:

| Metric | Value |
|---|---|
| **Image-only accuracy** | **66.14%** (macro-F1 60.0) |
| +metadata fusion | 79.31% (macro-F1 73.4) — *statistical, optimistic; banked already* |
| OOD abstention | 3.45% |

Per-class **image-only** recall (this is the real target):

| Class | recall | test n | note |
|---|---|---|---|
| Melanoma | 78.6% | **14** | n too small — number is noise, not signal |
| Basal cell carcinoma | 71.2% | 250 | malignant, well-supported |
| Actinic keratoses | 65.5% | 229 | |
| **Melanocytic nevi** | **57.6%** | 85 | weak |
| **Benign keratosis** | **56.7%** | 60 | weakest |

### Two facts that should drive everything below

1. **The encoder was never adapted to the clinical modality.** `raresight_nblk4mix` was
   fine-tuned on *dermoscopy* (HAM) + DermaMNIST for the resolution work. The clinical path
   reuses it frozen and only rebuilds *prototypes* from PAD images. **This is the single
   biggest untapped lever** — but see the risk note (it's contained).

2. **The clinical path is fully parallel.** Separate prototypes, separate OOD/calibration,
   separate routing. **Clinical-only changes cannot touch the validated dermoscopy results or
   the locked thesis numbers.** This is what lets us be more aggressive here than the
   "added capacity hurts" history (fusion_net, CoCoOp) would otherwise allow — those negatives
   were on the *dermoscopy* path; the clinical path is a clean, separate experiment.

### The "all classes" ask splits into two problems — they are not the same

- **(A) Lift the 5 measurable classes** (weakest: bkl 56.7, nv 57.6). Method levers below.
- **(B) Expand coverage** — and here there is a **hard data ceiling you must state, not paper over:**
  - **SCC is dropped entirely** today (no HAM mapping) → a real malignancy is silently ignored.
  - **df / vasc have zero clinical data** in PAD → cannot be classified on the clinical path at all.
  - **mel test n=14** → any per-class mel claim is noise. Don't over-report it.
  Fixing coverage needs *new data*, not a better method. That's the dataset section.

---

## 2. Ranked options — lift the 5 measurable classes (method, low data, frozen encoder)

Ranked by **regime fit** (small data, our own "capacity hurts" prior) and cost, *not* by the
gains the papers report (those are natural-image / different-taxonomy numbers — they won't
transfer 1:1; treat each as "candidate lever + why it fits + rough cost," let experiments decide).

### Tier 1 — cheap, regime-aligned, do first

| # | Lever | Why it fits | Cost / risk | Targets |
|---|---|---|---|---|
| 1 | **Color-constancy preprocessing** (Shades-of-Gray / Gray-World normalization before encoding) | Clinical phone photos have huge illumination/white-balance variance; SoG is the *classic* derm-photo win and is **training-free**. Applied identically at proto-build and inference. | ~1 hr; near-zero risk (preprocessing only) | All 5, esp. lighting-sensitive bkl/nv |
| 2 | **Test-time augmentation** on the clinical path | You already measured **+1pt on dermoscopy** with TTA. Free to port. | ~1 hr; zero risk | All 5 |
| 3 | **Tip-Adapter-F / Proto-Adapter** (cache-model adapter on *frozen* features) | **This is literally your own TODO** ("Later: Tip-Adapter, LoRA"). It extends your prototype approach with a key-value cache over PAD-train, no encoder touch. Proto-Adapter = constant-size variant (one layer), avoids Tip's cache-grows-with-data problem and reportedly beats it. | ~half day; low risk (frozen encoder, tiny adapter — but watch overfit on ~1.6k imgs, use the val/test protocol you already trust) | All 5; weak classes most |

These three are the "do these regardless" set: cheap, contained, and each independently
defensible as a thesis ablation (incl. honest nulls if they don't move).

### Tier 2 — higher cost, now *contained* by the parallel-path point

| # | Lever | Why it fits | Cost / risk | Targets |
|---|---|---|---|---|
| 4 | **Clinical-specific encoder adapter** — LoRA or last-N ViT blocks fine-tuned **on PAD-train**, clinical path only | The untapped lever from §1. You already have the multi-res fine-tune machinery (`finetune_multires.py`) and a working GPU (RTX 3050, 6GB — verified ~900 MB at bs16). Because it writes a *clinical-only* encoder slot, dermoscopy stays byte-identical. | 1–2 days; **medium** risk — ~1.6k imgs is small, real overfit danger; mitigate with LoRA (few params) + your val→test selection. Honest-negative outcome is itself a thesis finding. | All 5 |
| 5 | **Swap/ensemble a clinical-derm foundation model for the clinical path** — **PanDerm** (Nature Medicine 2025) is the strongest clinical-modality encoder found; explicitly covers the *clinical* image modality (BiomedCLIP is biomedical-general). Use as a drop-in clinical encoder, or ensemble its embedding with BiomedCLIP's. | Directly attacks the "encoder isn't clinical" root cause without you training the encoder. | 1–3 days integration (new preprocess, embedding dim, weights/license check); medium risk; the *biggest potential lift* | All 5 + better df/vasc/SCC representation |

> Note on DermLIP / MONET: DermLIP (Derm1M) reports beating biomedical CLIP zero-shot; MONET
> is dermatology-CLIP. PanDerm's own paper found CLIP-large beat dermatology-specific CLIPs at
> their scale — so **don't assume** a derm-CLIP beats BiomedCLIP here; benchmark before committing.

---

## 3. Expand coverage — recover dropped classes (needs new data, not a better method)

This is where the **"rare/underrepresented dermatological conditions in low-resource settings"**
framing gets real teeth, and it doubles as a **skin-tone fairness** story (PAD is one
Brazilian source; these add diversity).

| Action | What it buys | Cost |
|---|---|---|
| **Recover SCC** as a clinical-only 6th class (PAD *has* SCC, you currently drop it) | Stops silently ignoring a malignancy; cheap — data is already on disk, just unmap the drop | low (label + proto build) |
| **Add SCIN** (Google, ~5k crowdsourced clinical photos, diverse skin tones, derm labels) | More clinical support + fairness | med (taxonomy harmonization) |
| **Add DDI** (Diverse Dermatology Images — biopsy-confirmed, dark skin tones, rare+common) | Gold-standard labels, fairness, rare conditions | med |
| **Add Fitzpatrick17k** (16.5k clinical, 114 conditions, Fitzpatrick I–VI) | Could give **df / vasc** real clinical support + broad coverage | high (114→7 mapping, label noise) |
| **Add SD-198 / SD-260** (6.5k clinical, 198 categories) | Broad clinical coverage | high (mapping) |

**Hard caveat to state in the thesis:** every one of these has a *different taxonomy* than HAM's
7 classes. The win is real but the cost is **label harmonization** (and label noise, esp.
Fitzpatrick17k). Scope it: SCC-recovery + one dataset (SCIN or DDI) is a realistic thesis-sized
increment; "add all of them" is not.

---

## 3b. Tier-1 — RESULTS (run 2026-06-12, `scripts/tier1_clinical_experiments.py`)

Controlled isolation; rebuilt baseline reproduced the deployed **66.14% / mF1 60.03 EXACTLY**
(harness faithful). Clinical path only, image-only, PAD-TEST n=638. Prototypes + Mahalanobis
rebuilt under each transform (not just applied at test). Results → `checkpoints/tier1_clinical_results.json`.

| Config | Acc | mF1 | bkl | nv | Verdict |
|---|---|---|---|---|---|
| Baseline (deployed) | 66.14 | 60.03 | 56.7 | 57.6 | — |
| L1 color constancy | 65.05 | 59.62 | 55.0 | 56.5 | ❌ drop (−1.09) |
| **L2 TTA** | **66.61** | **61.39** | **60.0** | **60.0** | ✅ **keep** (+0.47 / +1.36 mF1) |
| L3 Tip-Adapter | 51.57 | 22.49 | 0.0 | 0.0 | ❌ drop (collapse to bcc 98%) |
| Combined (keepers) | 66.61 | 61.39 | 60.0 | 60.0 | = TTA only |

- **TTA kept** — marginal acc gain but real macro-F1 +1.36 and lifts the two weakest classes
  (bkl, nv +3.3 each). Training-free, no overfit risk.
- **Color constancy dropped** — Shades-of-Gray hurt ~1pt; re-illuminating moves inputs off the
  frozen BiomedCLIP manifold (it was trained on un-normalized photos). Clean negative.
- **Tip-Adapter dropped** — standard drop-in cache over the *imbalanced* full PAD-train collapses
  to the majority class (assumes balanced K-shot). Salvageable only with a class-balanced cache,
  which is no longer a cheap drop-in → deferred.

**PROMOTED TO LIVE (2026-06-12).** TTA wired into the deployed clinical path:
- `scripts/build_clinical_tta.py` rebuilt `clinical_prototypes.pt` + Mahalanobis/calib under TTA
  (verified PAD-TEST 66.61% / abstain 3.13%, reproduces the experiment); metadata-fusion tables
  preserved unchanged; `encoding="tta_flip_rot6"` flag added.
- `inference.py`: `_encode_tta()` + guarded re-encode inside the clinical branch (fires only when
  `clinical_serving["encoding"]=="tta_flip_rot6"`). Modality probe still uses the base embedding.
- Backups: `clinical_{prototypes,serving_params}_backup_20260612_pretta.pt` (rollback = copy back).
- Verified live `predict()`: clinical PAD images route clinical + TTA-encoded; dermoscopy (HAM/ISIC)
  still routes dermoscopy band-450, TTA NOT applied → dermoscopy path provably unchanged.
- Cost: ~6× encode per clinical-routed query (single forward → 6-view batch). Dermoscopy unaffected.

## 4. Recommendation (what I'd actually do, in order)

Given ~2 weeks to submission and the honest-negatives preference:

1. **Tier 1 all three** (color constancy → TTA → Proto-Adapter). Cheap, contained, each is a
   clean ablation row whether it wins or nulls. Targets the **66.1% image-only** number directly.
2. **Recover SCC** (cheap coverage win, removes an embarrassing gap).
3. **Then pick ONE Tier-2 swing** — I'd try **PanDerm as the clinical encoder** before training
   your own adapter: it attacks the root cause (non-clinical encoder) with no training risk, and
   if it underperforms BiomedCLIP that's a citable finding too. Fall back to LoRA-on-PAD if
   integration is too heavy.
4. **One new dataset (SCIN or DDI)** only if time allows — strong fairness/coverage story but
   the harmonization cost is real.

**Report image-only throughout** (the 79.3% is metadata-fused and already banked; the honest
lever is 66.1%). Keep dermoscopy artifacts untouched — every change above writes to clinical-only
files.

---

## Sources

- Tip-Adapter — [arXiv:2207.09519](https://arxiv.org/abs/2207.09519) · [code](https://github.com/gaopengcuhk/Tip-Adapter)
- Proto-Adapter (training-free, constant-size, beats Tip-Adapter) — [Sensors 2024 / PMC11175357](https://pmc.ncbi.nlm.nih.gov/articles/PMC11175357/)
- PanDerm — multimodal clinical-dermatology foundation model — [Nature Medicine 2025](https://www.nature.com/articles/s41591-025-03747-y) · [arXiv:2410.15038](https://arxiv.org/abs/2410.15038) · [code](https://github.com/SiyuanYan1/PanDerm)
- DermLIP / Derm1M — [arXiv:2503.14911](https://arxiv.org/html/2503.14911v1)
- Fitzpatrick17k — [dataset overview](https://www.emergentmind.com/topics/fitzpatrick-17k-dataset)
- DDI / SCIN / SD-198 — [Skin Type Diversity review, PMC11343783](https://pmc.ncbi.nlm.nih.gov/articles/PMC11343783/)
- Few-shot long-tail skin disease (meta/transfer) — [arXiv:2404.16814](https://arxiv.org/pdf/2404.16814)
