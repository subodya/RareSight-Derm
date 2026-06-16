"""Test the hypothesis that eval_results.json's 63.56% came from classifying test
queries against the DEPLOYED 20-shot M3 prototypes (disease_prototypes.pt), not
5-shot episodic prototypes.

Two evals against the deployed prototypes:
  (A) full 7-way: every test image classified against all 7 deployed prototypes
  (B) 5-way episodic: per episode, classify queries against the 5 relevant
      deployed prototypes (matches evaluate.py's 5-way framing but with 20-shot protos)

If either reproduces ~63.5% with class-0 LOW / nevi HIGH, the 63.56 origin is confirmed.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.data.dataset import EpisodicDermaMNIST
from PIL import Image
from src.data.preprocessing import load_ham10000

N_CLASSES = 7
CLASS_NAMES = {0:"Actinic keratoses",1:"Basal cell carcinoma",2:"Benign keratosis",
               3:"Dermatofibroma",4:"Melanoma",5:"Melanocytic nevi",6:"Vascular lesions"}
CKPT = "checkpoints/raresight_nblk4mix.pth"


def _norm(x):
    return x / x.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev)
    m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)
    m.eval()
    temp = m.temperature.item()

    protos = torch.load("src/app/assets/disease_prototypes.pt", map_location=dev)
    P = torch.stack([_norm(protos[c].to(dev)) for c in range(N_CLASSES)])  # (7,512)
    print(f"Deployed prototypes loaded (20-shot M3). temp={temp:.3f}\n")

    # Encode the full HAM10000 test split
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    paths, labels = load_ham10000(data_root=os.path.join(project_root, "data", "ham10000"),
                                  split="test", val_size=0.1, test_size=0.1, seed=42)
    labels = np.array([int(l) for l in labels])
    embs = []
    with torch.no_grad():
        for i in range(0, len(paths), 16):
            batch = torch.cat([m.preprocess(Image.open(p).convert("RGB")).unsqueeze(0)
                               for p in paths[i:i+16]]).to(dev)
            embs.append(_norm(m.backbone.encode_image(batch)).cpu())
    Q = torch.cat(embs)  # (Ntest,512)
    print(f"Encoded {len(Q)} test images.\n")

    # (A) full 7-way
    d = torch.cdist(Q, P.cpu())
    pred = (-d).argmax(1).numpy()
    acc7 = 100.0 * (pred == labels).mean()
    pc7 = {c: 100.0*((pred==c)&(labels==c)).sum()/max((labels==c).sum(),1) for c in range(N_CLASSES)}
    macro7 = np.mean([pc7[c] for c in range(N_CLASSES)])
    print(f"(A) FULL 7-way deployed-prototype eval:  overall={acc7:.2f}%  macro_recall={macro7:.2f}%")
    for c in range(N_CLASSES):
        print(f"      {CLASS_NAMES[c]:<22} recall={pc7[c]:.1f}%  (n={int((labels==c).sum())})")

    # (B) 5-way episodic against deployed protos
    ds = EpisodicDermaMNIST(split="test", augment=False)
    np.random.seed(42); torch.manual_seed(42)
    cor = tot = 0
    pc_cor = {c:0 for c in range(N_CLASSES)}; pc_tot = {c:0 for c in range(N_CLASSES)}
    with torch.no_grad():
        for _ in range(300):
            _, _, q_img, q_lbl, oc = ds.sample_episode(5, 5, 15, return_class_ids=True)
            oc = np.array(oc); lg = oc[q_lbl.numpy()]
            q_emb = _norm(m.backbone.encode_image(q_img.to(dev)))
            sub = P[oc]  # (5,512) deployed protos for this episode's classes
            pl = (-torch.cdist(q_emb, sub)).argmax(1).cpu().numpy()
            pg = oc[pl]
            cor += int((pg==lg).sum()); tot += len(lg)
            for c in lg: pc_tot[int(c)] += 1
            for p,t in zip(pg,lg):
                if p==t: pc_cor[int(t)] += 1
    acc5 = 100.0*cor/tot
    print(f"\n(B) 5-way episodic vs deployed 20-shot protos (seed42,300ep): overall={acc5:.2f}%")
    for c in range(N_CLASSES):
        if pc_tot[c]:
            print(f"      {CLASS_NAMES[c]:<22} recall={100.0*pc_cor[c]/pc_tot[c]:.1f}%")

    json.dump({"full7way": {"overall": round(acc7,2), "macro_recall": round(macro7,2),
                            "per_class": {str(c): round(pc7[c],1) for c in range(N_CLASSES)}},
               "episodic5way_deployedprotos": round(acc5,2)},
              open("checkpoints/diag_deployed_eval.json","w"), indent=2)
    print("\nSaved → checkpoints/diag_deployed_eval.json")


if __name__ == "__main__":
    main()
