# RareSight — Consolidated Results (defensible, reproducible)

All numbers on **HAM10000**, checkpoint `raresight_finetuned.pth`, deterministic
splits (stratified, seed 42). Regenerate via the scripts noted per section.

> **NOTE (2026-06-10):** §"Phase-2 — Multi-resolution encoder fine-tuning" replaced the deployed
> encoder (`raresight_finetuned.pth` is now the fine-tuned weights; the frozen original is backed up
> at `checkpoints/raresight_finetuned_backup_20260610.pth`). The RQ1/RQ2/RQ3 episodic sections below
> **predate Phase-2** and were measured on the *frozen* BiomedCLIP backbone.

> **DEPLOYMENT UPDATE (2026-06-09 session 2):** the served recipe is now **CoOp**, not M3.
> CoOp beats M3 in the deployed 7-way K=20 regime (acc 56.96 vs 51.57, macro-F1 47.39 vs
> 44.95, McNemar p<1e-4; **with metadata: 62.80 / 50.01 vs 57.71 / 47.35**). Calibration
> improved (ECE 0.048→0.032); OOD pinned to Mahalanobis (validated webp/PAD safety
> preserved). M3 backed up in `src/app/assets/_m3_deploy_backup_20260609/`. CoCoOp was
> tried and is a clean negative. Full change-log: `thesis/SESSION_coop_deploy_and_cocoop.md`.

> **DEPLOYMENT UPDATE (2026-06-09 session 3) — RESOLUTION + MODALITY routing:** the served
> dermoscopy prototypes were full-res-only and FAILED on low-res uploads (28×28 derma_samples:
> 17% acc, 100% OOD-rejected). Fixed with resolution-banded prototypes + native-res routing
> (28px 27.8→52.7% acc, abstain ~100→0%; full-res unchanged) and a **modality router** that
> sends smartphone photos to the clinical path (PAD 23.2→**64.3%**; dermoscopy preserved).
> **The §"Far-OOD webp flagged" claim below is SUPERSEDED:** that was *blanket rejection of all
> clinical photos*. With the clinical path, the webp (a real dermatofibroma clinical photo) now
> routes clinical and **REFERS** (df/vasc are a documented clinical-data gap → flagged, not
> silently mislabelled); only non-images/garbage abstain (noise ood −544, grey −526 ≪ τ).
> Citable low-res headline: DermaMNIST-TEST **39.0%** deployed *(SUPERSEDED 2026-06-10 → 62.39%, see
> §Phase-2 below)*. Full change-log: `thesis/SESSION_resolution_diagnosis.md`.

> **DEPLOYMENT UPDATE (2026-06-10 session 5) — PHASE-2 FINE-TUNED ENCODER (now deployed):** the frozen
> low-res ceiling was lifted by fine-tuning the **visual encoder's last 4 ViT blocks** (multi-res
> scale-augmentation + a **real-DermaMNIST-28 training mix**), then rebuilding every serving artifact
> on the adapted encoder (modality gap recomputed, global β re-tuned 0.75→0.65). **This is the
> project's first TRAINED-encoder deployment** — the CoOp text / M3 blend stack is retained, but the
> *image* tower is no longer frozen. Citable low-res headline jumps **DermaMNIST-TEST 38.8 → 62.39%
> (+23.6pp)**; full-res preserved/up; low-res referral slashed (~70→~13%); calibration improved;
> clinical path rebuilt (PAD 64.3→66.1%). Prior (frozen) deployment backed up at
> `src/app/assets_backup_20260610_pre_nblk4mix/` + `checkpoints/raresight_finetuned_backup_20260610.pth`.
> Detail in §Phase-2 below; session log in `memory/raresight-thesis-progress.md`.

## Phase-2 — Multi-resolution encoder fine-tuning (DEPLOYED 2026-06-10)
Fine-tune `src/training/finetune_multires.py` (`--nblocks 4 --mix-real-dm 0.5`, seeds 42/123, AMP
fp16, layer-wise LR decay; last 4 of 12 ViT blocks ≈ 29M trainable params; RTX 3050 6GB). Model
selection on the **deployment metric** — rebuild support-mean prototypes → cosine-NN on real
DermaMNIST-**val**@28, with a full-res HAM@224 guard, reported on **test** (no selection leak).
Deployed via `build_adapted_ckpt.py` → `rebuild_blend_newgap.py` → `build_serving_multires.py` +
`patch_band28_mix.py` + `build_modality_probe.py` + `build_clinical_path.py` (env-parameterized
`RS_CKPT/RS_OUT_DIR/RS_BLEND`). Measured with `src/app/tools/eval_deployed.py` (image-only path;
controlled frozen-vs-fine-tuned staging — the frozen staging reproduced the live 38.80 *exactly*).

**Deployed per-band TEST accuracy (image-only; the app's metadata fusion adds ~+5–6pp on top):**

| Band | Frozen (old deploy) | Fine-tuned (deployed) | Δ |
|---|---|---|---|
| 28px | 48.0 | 61.0 | +13.0 |
| 56px | 54.4 | 60.1 | +5.7 |
| 112px | 57.0 | 58.0 | +1.0 |
| 224px (full-res) | 55.1 | 56.0 | +0.9 |
| 450px (full-res) | 54.4 | 57.3 | +2.9 |
| **DermaMNIST-test (2005 imgs, citable)** | **38.8** | **62.39** | **+23.6** |

> **No full-res regression** — accuracy rose at *every* band (the guard never bound; the proxy's
> full-res gain held). Low-res **referral** collapsed (28px 69.9→11.8%, DM-28 72.7→18.2%): the app
> went from punting ~70% of low-res inputs to committing on ~85% and being accurate when it does.
> **Calibration** improved (DM-28 ECE 0.117→0.026). **Modality routing** improved (real 28px
> dermoscopy mis-routed-to-clinical 11.4→5.7%; live spot-check: 94% route dermoscopy). **Clinical
> path** rebuilt on the new encoder: PAD-test 64.3→66.1%, melanoma recall 78.6%, abstain 3.5%.

**Per-class recall, real DermaMNIST-test@28 (frozen → deployed):**

| Class | n | Frozen | Deployed |
|---|---|---|---|
| nv (nevus) | 1341 | 47.1 | **75.5** |
| akiec | 66 | 13.6 | 40.9 |
| bkl | 220 | 15.0 | 34.1 |
| **mel (melanoma)** | 223 | 16.1 | 28.7 |
| bcc | 103 | 34.0 | 37.9 |
| df | 23 | 39.1 | 34.8 |
| vasc | 29 | 86.2 | 86.2 |

> **Honest per-class read.** The +23.6pp overall is **nv-dominated** (nv = 67% of the test set; its
> +28pp drives most of the headline). **Melanoma improved +12.6pp but remains the weakest major class
> (28.7%)** — the key clinical limitation; the referral gate flags many uncertain mel cases. **bcc**
> peaked at the old-gap/β=0.75 config (52%) and the re-tuned β=0.65 traded some back (37.9%) — a
> malignant-class trade-off worth watching (small n=103). Report macro/per-class, not just overall.

**What was tried (kept for the thesis — two clean negatives):**
- **Block-count sweep {0,2,3,4}:** capacity *helps* monotonically here — under scale-aug + real data,
  with no full-res cost. **Contradicts the earlier "trainable capacity hurts" thread** (CoOp>CoCoOp,
  fusion-net). nblk4 best, stable across two seeds.
- **Real-DermaMNIST-28 training mix (#2):** the decisive lever (+3.1pp over plain scale-aug on the
  proxy) — closes the synthetic/real gap that scale-aug alone plateaued on.
- **Degradation-realistic low-res aug (#3) — NEGATIVE** (−0.65pp): random-kernel + JPEG + sensor-noise
  synthetic degradation ≠ real sensor statistics. Faking low-res does not substitute for real data.
- **Per-band β re-tune on the FROZEN encoder — mostly NEGATIVE / off-target** (28px −0.57, real DM
  +0.1; only 112px +5.57). The "β too high at low-res" prior was *not* supported. NB re-tuning β on the
  **fine-tuned** encoder (after recomputing the gap) *did* help (0.75→0.65, +2.6pp) — encoder-specific.

**Caveats / follow-ups:** the eval is image-only (metadata fusion adds ~+5–6pp on top); the single-res
`disease_prototypes.pt`/`serving_params.pt` were left on the frozen encoder (never hit in the main path
— `_select_band` always routes to a band when multi-artifacts exist — only the demo *refinement* mode
mixes them). Artifacts: `checkpoints/finetune_multires/nblk4_seed*_mix0.5_best.pth`,
`thesis/eval_deployed_assets_staging_*.json`, `thesis/beta_retune_results.json`.

## Headline: few-shot accuracy (RQ1) — 5-way 5-shot episodic
`src/training/evaluate.py` (EVAL_MODE=m3, 600 ep, seed 42)

| Method | Accuracy | Notes |
|---|---|---|
| Zero-shot BiomedCLIP | 42% | project baseline |
| Standard ProtoNet (ResNet-50) | 61% | project baseline |
| MLP-fusion `forward()` (old deployed) | 54.7% | **hurts** — distorts aligned space |
| Image-only prototypes (M0) | 57.4% | |
| **M3 aligned CuPL+gap blend (training-free)** | **60.9%** | macro-F1 61.3, ECE 0.316 (episodic) |
| **CoOp learned prompts (trained, best accuracy, NOW DEPLOYED)** | **64.4%** | see CoOp section; +3.7 vs M3 on matched episodes |
| Full-data supervised (7-way) | 78% | upper bound, *not* a rival |

> The 63.56% once reported was a stale artifact (test queries vs 20-shot prototypes,
> a different protocol), not the 5-shot `forward()` — see memory `multimodal-fusion-findings`.

## RQ1 — shot curve (K = 1 / 5 / 10), 5-way episodic
`src/training/eval_shot_curve.py` (seed 42, val 200 / test 300 ep; β,λ re-tuned on val
per K; modality gap recomputed per K). `checkpoints/shot_curve_results.json`.

| K (shots) | Zero-shot | Image-only | M3 (CuPL+gap) | CoOp | tuned β | M3−img | CoOp−img |
|---|---|---|---|---|---|---|---|
| 1  | 33.1 | 41.9 | 52.2 | **62.0** | 0.55 | +10.3 | +20.1 |
| 5  | 33.0 | 57.4 | 60.9 | **64.2** | 0.75 | +3.6  | +6.9  |
| 10 | 34.2 | 62.4 | 63.7 | **65.2** | 0.85 | +1.3  | +2.8  |

> **Headline finding (data, not tuned): text helps most when images are scarce.** The
> M3−image and CoOp−image gains grow monotonically as K shrinks (+1.3→+3.6→+10.3 for M3;
> +2.8→+6.9→+20.1 for CoOp), and the val-tuned image weight β drops 0.85→0.75→0.55 — at
> 1-shot the noisy image prototype leans on the vision-language prior. CoOp wins at every K
> and dominates at K=1 (+20.1 over image-only). This directly supports the "rare disease /
> few images" motivation: the contribution matters most exactly where data is thinnest.
> Zero-shot (cosine to CuPL text, no support) is a flat ~33% anchor — few-shot clears it
> even at K=1. All methods monotone in K. ProtoNet baseline deferred (separate ResNet-50
> harness); prior single-point ProtoNet = 61% (5-way 5-shot).

## RQ2 — Multimodal prototype ablation (training-free)
`src/training/eval_multimodal_tier1.py` (seed 42, 300 test ep)

| Method | Test acc | Δ vs image-only |
|---|---|---|
| M0 image-only | 57.37 | — |
| M1 blend (orig text) | 59.02 | +1.65 |
| M2 blend (CuPL) | 59.29 | +1.92 |
| **M3 blend (CuPL + modality-gap)** | **60.93** | **+3.56** |
| M4 per-class β | 55.78 | −1.59 (overfit; honest negative) |
| M5 Tip-Adapter logit fusion | 57.25 | −0.12 (text zero-shot too weak) |

## RQ2 capstone — CoOp learned prompts (trained, beats M3)
`src/training/train_coop.py` (COOP_MODE=indist, seed 42, 1500 train ep, 300 test ep,
5-way 5-shot). Trains ONLY a 4-token context vector (3,072 params); BiomedCLIP
backbone and β/λ/gap fixed at M3 values; CuPL distillation regulariser. Full run,
validation ran (best_val 64.29 — not the earlier `best_val=-1` smoke artifact).

| Method (same 300 episodes) | Test acc | Δ vs image-only |
|---|---|---|
| Image-only | 57.29 | — |
| M3 CuPL+gap blend | 60.71 | +3.42 |
| **CoOp learned context** | **64.44** | **+7.15** (+3.73 vs M3) |

> CoOp is the **best 5-way 5-shot accuracy** in the project. M3 stays the **deployed**
> recipe (training-free, no per-deployment tuning); CoOp is the trained-prompt capstone
> showing learned context beats hand-built CuPL prompts. Artifacts:
> `checkpoints/coop_indist_results.json`, `coop_indist_ctx.pt`, `coop_indist_full.log`.
> NOVEL 2-way {df, vascular} CoOp is near-ceiling (~96.6%, non-discriminative) and was
> a smoke run (`best_val=-1`) — a 3-way novel split is the real generalization test (pending).

## Novel-class generalization — 3-way leave-classes-out
`COOP_MODE=novel COOP_NOVEL=1,3,4 src/training/train_coop.py` (hold out BCC + melanoma +
dermatofibroma; CoOp context trained ONLY on the 4 base classes {0,2,5,6}, then tested
3-way on the held-out classes). Seed 42, 300 test ep. `checkpoints/coop_novel_134_results.json`.

| Method (3-way novel, chance = 33%) | Test acc | Δ vs image-only |
|---|---|---|
| Image-only | 68.92 | — |
| **M3 CuPL+gap blend** | **70.77** | **+1.85** |
| CoOp learned (trained on base classes) | 70.30 | +1.38 (−0.47 vs M3) |

> **Honest generalization finding.** (1) Well above chance (70% on 3-way) and *discriminative*
> — unlike the old 2-way {df,vascular} split (~96%, near-ceiling, useless). (2) The
> training-free text blend (M3) **still helps on unseen classes** (+1.85). (3) CoOp's
> in-distribution edge (+3.7 over M3) does **NOT transfer**: its context, tuned on the base
> classes, lands slightly below generic CuPL text on novel classes. This is the defensible
> RQ answer to "is this really few-shot generalization?".
>
> **UPDATE (2026-06-09 session 2):** a per-episode paired t-test shows the CoOp−M3 novel
> difference (−0.47) is **NOT significant** (p=0.14, `coop_novel_sig.json`) — so CoOp does
> NOT generalize *worse* than M3; they tie on novel. A rigorous base-to-new eval
> (`cocoop_compare.json`) gives CoOp the **best** harmonic mean (72.56 > M3 70.81): CoOp
> ties M3 on novel AND wins on base. So the earlier "M3 is the more robust generalizer"
> framing is **withdrawn** — read (3) as "CoOp's *extra* in-dist edge doesn't transfer to
> novel, but it does not regress there either." (CoCoOp, by contrast, *does* overfit base
> and regress on novel — see `thesis/SESSION_coop_deploy_and_cocoop.md`.)
>
> **Limitation (state it):** the frozen backbone (`raresight_finetuned.pth`) was meta-trained
> on all 7 classes, so the *backbone* has seen these classes — this experiment isolates whether
> the **learned prompt context** generalizes to held-out classes, not full backbone-level
> novelty (which would need a backbone retrain on base classes only). image-only/M3 numbers
> are therefore "novel for the trainable surface," not novel for the representation.

## Structured patient metadata fusion (NEW — strongest novel result)
`src/training/_diag_meta_fusion.py`. Age/sex/site fused as class-conditional
likelihood `log P(meta|c)` (no prevalence prior); α tuned on val macro-F1.
**Reported on macro-F1** (balanced) so the gain is genuine diagnosis, not nevi regression.

| Protocol | Variant | Overall | **Macro-F1** | Balanced acc |
|---|---|---|---|---|
| Deployed 7-way (α=0.25) | image-only | 51.6 | 44.9 | 57.6 |
| | **+ metadata** | 57.7 | **47.4 (+2.4)** | 62.1 |
| Balanced 5-way (α=0.5) | image-only | 56.8 | 57.2 | 56.8 |
| | **+ metadata** | 65.0 | **63.9 (+6.7)** | 64.4 |

> In the *balanced* protocol the prevalence prior cancels, so +6.7 macro-F1 proves
> conditional age/sex/site structure improves diagnosis (e.g. actinic-keratoses ↔
> older + sun-exposed site). This is the defensible viva centerpiece.

## RQ3 — Calibration & open-set (deployed app)
`src/app/tools/build_serving_artifacts.py` (fit on val, reported on held-out test)

| Metric | Value | Method |
|---|---|---|
| ECE before calibration | 0.30–0.31 | — |
| **ECE after temperature scaling** | **0.048** | meets <0.10 target; T per regime (img 0.16 / +meta 0.23) |
| Open-set AUROC (near-OOD, leave-one-class-out) | 0.64 (Mahalanobis) | hard: held-out classes are same-domain |
| Far-OOD detection | webp (off-modality) flagged unknown | ood −83 ≪ τ −30 (2% false-abstain) |

## Manual UI verification — OOD guard + few-shot recovery (CORRECTED 2026-06-15)
> **CORRECTION (2026-06-15):** the earlier "Dermatofibroma 99.6% few-shot recovery" on the
> off-modality webp was an **artifact of an OOD bug**, not a real result, and is withdrawn.
> After Phase-2 the single-res refinement artifacts (`disease_prototypes.pt`/`serving_params.pt`)
> were left on the pre–Phase-2 encoder, so *every* refined query scored as OOD and the (also
> stale) competing class prototypes sat artificially far away — the user's support prototype
> won trivially, inflating confidence. The artifacts were rebuilt on the deployed encoder
> (main dermoscopy/clinical paths verified unchanged, 9/9 identical). With aligned artifacts
> the cross-domain webp recovery **does not reproduce** — the off-manifold webp now correctly
> **OOD-flags** instead. The honest few-shot effect is a *modest in-domain assist* (below).
> Source of record: `DEMO_VIDEO_RUNBOOK.md` (verified live 2026-06-15).

**OOD refusal (still valid, now stronger).** Off-modality clinical wide-field photo
`dermafibroma-sample.webp` (lesion ~10% of frame; model trained on dermoscopy) is correctly
**flagged unknown / referred** in the patient-scan path — post–Phase-2 it sits off the
encoder's manifold, so the Mahalanobis rejector catches it (far-OOD: webp ood −83 ≪ τ −30,
~2% false-abstain). The system does **not** confidently misdiagnose an out-of-distribution
image. (Matches `verify_serving.log`.)

**Few-shot recovery (honest in-domain case, verified live 2026-06-15).** Teaching a weak,
heterogeneous class from 5 examples flips a wrong benign-vs-benign call to the correct one:
query `QUERY_bkl_run_this_first.jpg` → **before:** Melanocytic nevi 0.39 (wrong; true class
is benign keratosis), not referred → after teaching 5 "Benign keratosis" support images →
**after:** Benign keratosis 0.61 (correct), top-3 = [bkl 0.61, nv 0.27, mel 0.07], not
flagged, not referred. Files: `demo_test_samples/fewshot_bkl/`; probe `scripts/fewshot_demo_probe.py`.

This is the defensible open-set/domain-shift story: **refuse OOD in zero-shot** (the webp),
and **modestly recover an in-domain weak class with few-shot exemplars** — narrate the
few-shot effect as a real clinic-side assist (~0.4→0.6), not a dramatic flip.

## Clinical app features (status)
- **Metadata fusion** — live, validated above.
- **Calibration** — live temperature scaling, real ECE (replaced hardcoded 0.07).
- **Open-set rejection** — Mahalanobis abstain @2% false-abstain; modality guard.
- **Clinical notes** — BiomedCLIP text-match, gated; *demo* (HAM10000 has no notes).
- **PDF export** — deferred.

## Known limitations (state proactively)
- **Melanoma at low resolution is the weakest major class** (DermaMNIST-test@28 recall 28.7% after
  Phase-2, up from 16.1% but still low). Malignant; the referral gate flags uncertain cases, but
  recall is the number to watch. The Phase-2 low-res headline (+23.6pp) is also **nv-dominated** — read
  per-class, not just overall.
- Near-OOD detection is weak (0.64); the rejector targets far-OOD / wrong-modality.
- Clinical-note influence is unvalidated (no notes in data) and a confident-but-wrong
  note can still nudge ranking (β=0.5, gated, demo-only).
- HAM10000 `lesion_id` appears in both train and test (not used as a feature).
- Novel-class 2-way {df, vascular} is near-ceiling (~96%) — not discriminative; a
  3-way novel split or cross-dataset eval would be a stronger generalization test.

_Reproduce: see `memory/` (multimodal-fusion-findings, clinical-app-features,
raresight-next-tasks) and the scripts referenced per section._
