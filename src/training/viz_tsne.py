"""
t-SNE visualization + silhouette score of the BiomedCLIP image-embedding space (RO8).

Encodes a class-balanced sample of the HAM10000 TEST split (image-only embeddings),
projects to 2-D with t-SNE for a qualitative figure, and reports the silhouette score on
the 512-D normalized embeddings under cosine distance (the meaningful separability number;
the 2-D silhouette is reported too for reference).

Honest framing: on raw 7-way dermoscopy embeddings the silhouette lands well below the
proposal's 0.5 target — the classes genuinely overlap in vision space, which is *why* the
task is hard and motivates the text/metadata fusion contributions. We report the number as
measured; we do not tune the space to clear 0.5.

Run:  conda run -n raresight python src/training/viz_tsne.py
Env:  TSNE_PER_CLASS (default 200 images/class)
Out:  figures/tsne_embeddings.png , checkpoints/tsne_silhouette.json
"""

import sys, os, json, numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST

SEED, N_CLASSES = 42, 7
PER_CLASS = int(os.environ.get("TSNE_PER_CLASS", "200"))
CKPT = "checkpoints/raresight_nblk4mix.pth"
FIG  = "figures/tsne_embeddings.png"
OUT  = "checkpoints/tsne_silhouette.json"
CLASS_NAMES = {0:"Actinic keratoses",1:"Basal cell carcinoma",2:"Benign keratosis",
               3:"Dermatofibroma",4:"Melanoma",5:"Melanocytic nevi",6:"Vascular lesions"}
RARE = {3, 6}


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()

    ds = EpisodicDermaMNIST(split="test", augment=False)
    rng = np.random.default_rng(SEED)

    embs, labels = [], []
    print(f"Encoding up to {PER_CLASS} test images/class...")
    with torch.no_grad():
        for c in sorted(ds.class_to_indices.keys()):
            idx = np.array(ds.class_to_indices[c])
            pick = rng.choice(idx, size=min(PER_CLASS, len(idx)), replace=False)
            imgs = torch.stack([ds._load_image(int(i)) for i in pick]).to(dev)
            for i in range(0, len(imgs), 64):
                embs.append(_norm(m.backbone.encode_image(imgs[i:i+64])).cpu().numpy())
            labels += [c] * len(pick)
    X = np.concatenate(embs); y = np.array(labels)
    print(f"  encoded {len(y)} embeddings, dim={X.shape[1]}")

    sil_hi = float(silhouette_score(X, y, metric="cosine"))
    print(f"Silhouette (512-D, cosine): {sil_hi:.3f}   (proposal target 0.5)")

    print("Running t-SNE (this takes a minute)...")
    Z = TSNE(n_components=2, init="pca", perplexity=30, random_state=SEED,
             metric="cosine").fit_transform(X)
    sil_2d = float(silhouette_score(Z, y))

    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for c in sorted(set(y.tolist())):
        msk = y == c
        plt.scatter(Z[msk, 0], Z[msk, 1], s=10, alpha=0.6, color=cmap(c),
                    label=f"{'★ ' if c in RARE else ''}{CLASS_NAMES[c]}")
    plt.title(f"BiomedCLIP image embeddings (HAM10000 test) — t-SNE\n"
              f"silhouette: {sil_hi:.3f} (512-D cosine), {sil_2d:.3f} (2-D)")
    plt.legend(loc="best", fontsize=8, framealpha=0.9)
    plt.xticks([]); plt.yticks([]); plt.tight_layout()
    os.makedirs(os.path.dirname(FIG), exist_ok=True)
    plt.savefig(FIG, dpi=150)
    print(f"Saved figure → {FIG}")

    json.dump({"silhouette_512d_cosine": round(sil_hi, 4),
               "silhouette_2d_tsne": round(sil_2d, 4),
               "n_per_class": PER_CLASS, "n_total": int(len(y)), "seed": SEED,
               "proposal_target": 0.5,
               "note": "Raw 7-way image embeddings overlap (silhouette < 0.5); this "
                       "motivates the text/metadata fusion contributions."},
              open(OUT, "w"), indent=2)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
