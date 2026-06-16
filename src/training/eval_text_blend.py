"""
RQ2 (corrected) — training-free CLIP-space text blend.

In a contrastive VLM, image and text embeddings live in the SAME aligned space,
so the principled way to inject text into a prototype is a convex blend there —
not a learned MLP on concatenated features (which our ablation showed degrades
the embedding):

    proto_c = normalize( beta * image_proto_c + (1 - beta) * text_emb_c )

where image_proto_c = mean of K support image embeddings (class c)
      text_emb_c    = normalized encode_text(class description)

beta is swept on the VAL split and the single best beta is reported on TEST.
Endpoints are references:  beta=1.0 -> image-only,  beta=0.0 -> text-only.

    conda run -n raresight python src/training/eval_text_blend.py

Decision rule: if best-beta TEST acc beats the image-only (beta=1.0) number, the
multimodal thesis is rescued with a cleaner, training-free method. If not, text
does not add on this data -> report the honest negative and reframe RQ2.
"""

import sys, os, json, time, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

N_WAY, K_SHOT, N_QUERY = 5, 5, 15
VAL_EPISODES  = int(os.environ.get("BLEND_VAL_EP",  "200"))
TEST_EPISODES = int(os.environ.get("BLEND_TEST_EP", "300"))
SEED = 42
N_CLASSES = 7
CKPT = "checkpoints/raresight_nblk4mix.pth"
OUT  = "checkpoints/text_blend_results.json"
DESC = os.path.join(os.path.dirname(__file__), "../../src/app/class_descriptions.json")
BETAS = [round(b, 2) for b in np.linspace(0.0, 1.0, 11)]   # 0.0 .. 1.0
CLASS_NAMES = {0:"Actinic keratoses",1:"Basal cell carcinoma",2:"Benign keratosis",
               3:"Dermatofibroma",4:"Melanoma",5:"Melanocytic nevi",6:"Vascular lesions"}
RARE = {3, 6}


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    desc = json.load(open(os.path.abspath(DESC)))
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()
    temp = m.temperature.item()

    def encode_text_classes(orig_cls):
        texts = [desc.get(str(c), "dermoscopy image of a skin lesion") for c in orig_cls]
        toks = m.tokenizer(texts, padding="max_length", truncation=True,
                           max_length=256, return_tensors="pt")["input_ids"].to(dev)
        return _norm(m.backbone.encode_text(toks))            # (N_WAY, 512)

    def run_split(ds, n_ep, betas, track_per_class=False):
        """Returns {beta: overall_acc} and optional per-class dict for each beta."""
        correct = {b: 0 for b in betas}
        per_cls = {b: {c: [0, 0] for c in range(N_CLASSES)} for b in betas}  # [correct,total]
        tot = 0
        np.random.seed(SEED); torch.manual_seed(SEED)
        with torch.no_grad():
            for _ in range(n_ep):
                s_img, _, q_img, q_lbl, orig = ds.sample_episode(
                    N_WAY, K_SHOT, N_QUERY, return_class_ids=True)
                s_img, q_img = s_img.to(dev), q_img.to(dev)
                q_lbl = q_lbl.cpu().numpy()
                orig_arr = np.array(orig)
                labels_g = orig_arr[q_lbl]

                q_emb = _norm(m.backbone.encode_image(q_img))
                img_proto = _norm(_norm(m.backbone.encode_image(s_img)).view(N_WAY, K_SHOT, -1).mean(1))
                txt_emb = encode_text_classes(orig)            # (N_WAY,512)

                tot += len(labels_g)
                for b in betas:
                    proto = _norm(b * img_proto + (1.0 - b) * txt_emb)
                    pred_l = torch.softmax(-torch.cdist(q_emb, proto) * temp, 1).argmax(1).cpu().numpy()
                    pred_g = orig_arr[pred_l]
                    correct[b] += int((pred_g == labels_g).sum())
                    if track_per_class:
                        for pg, tg in zip(pred_g, labels_g):
                            per_cls[b][int(tg)][1] += 1
                            if pg == tg:
                                per_cls[b][int(tg)][0] += 1
        acc = {b: round(100.0 * correct[b] / tot, 2) for b in betas}
        return acc, per_cls

    print(f"Device {dev} | temp={temp:.3f} | betas={BETAS}")
    t0 = time.time()

    print(f"\nSweeping beta on VAL ({VAL_EPISODES} episodes)...")
    val_ds = EpisodicDermaMNIST(split="val", augment=False)
    val_acc, _ = run_split(val_ds, VAL_EPISODES, BETAS)
    best_beta = max(val_acc, key=val_acc.get)
    print("  VAL acc by beta:")
    for b in BETAS:
        flag = "  <- best" if b == best_beta else ""
        print(f"    beta={b:.1f}  {val_acc[b]:6.2f}%{flag}")

    print(f"\nReporting on TEST ({TEST_EPISODES} episodes) at best_beta={best_beta} "
          f"plus endpoints...")
    test_betas = sorted(set([best_beta, 0.0, 1.0]))
    test_ds = EpisodicDermaMNIST(split="test", augment=False)
    test_acc, test_pc = run_split(test_ds, TEST_EPISODES, test_betas, track_per_class=True)

    img_only = test_acc[1.0]
    text_only = test_acc[0.0]
    blended = test_acc[best_beta]
    print(f"\n  TEST  image_only (beta=1.0) : {img_only:.2f}%")
    print(f"  TEST  text_only  (beta=0.0) : {text_only:.2f}%")
    print(f"  TEST  blend      (beta={best_beta}) : {blended:.2f}%")
    print(f"  Delta blend - image_only    : {blended - img_only:+.2f} pts")

    print(f"\n  Per-class TEST acc (rare *):  {'img':>8}{'blend':>8}")
    per_class_out = {}
    for c in range(N_CLASSES):
        ci, ti = test_pc[1.0][c], test_pc[best_beta][c]
        if ti[1] == 0:
            continue
        a_img = round(100.0 * ci[0] / ci[1], 1)
        a_bl = round(100.0 * ti[0] / ti[1], 1)
        per_class_out[c] = {"image_only": a_img, "blend": a_bl}
        mark = "*" if c in RARE else " "
        print(f"  {mark}{CLASS_NAMES[c]:<24}{a_img:>7.1f}%{a_bl:>7.1f}%")

    res = {
        "checkpoint": CKPT, "seed": SEED,
        "val_episodes": VAL_EPISODES, "test_episodes": TEST_EPISODES,
        "val_acc_by_beta": val_acc, "best_beta": best_beta,
        "test": {"image_only": img_only, "text_only": text_only,
                 f"blend_beta_{best_beta}": blended,
                 "delta_blend_minus_image": round(blended - img_only, 2)},
        "test_per_class": {str(c): per_class_out[c] for c in per_class_out},
        "elapsed_min": round((time.time() - t0) / 60, 2),
    }
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"\nSaved -> {OUT}   ({res['elapsed_min']} min)")


if __name__ == "__main__":
    main()
