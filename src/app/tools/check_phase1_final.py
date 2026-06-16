"""
Final Phase-1 measurements (advisor): (1) does the far-OOD df webp REFER (entropy) or is it
confidently wrong? (2) deployed DermaMNIST-TEST headline through the FULL routing path
(resolution band-28 + modality probe), since ~11% now routes clinical.
"""
import os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "backend")))
import inference as inf
sys.path.insert(0, inf.PROJECT_ROOT)

res = inf.load_resources()

# ---- (1) webp: refer? confidence? ----
webp = os.path.join(inf.PROJECT_ROOT, "dermafibroma-sample.webp")
if os.path.exists(webp):
    o = inf.predict(Image.open(webp).convert("RGB"), res)
    print("webp (true class = dermatofibroma, unrepresentable on clinical path):")
    print(f"  modality={o['modality']} p_clin={o['p_clinical']} is_unknown={o['is_unknown']} "
          f"refer={o['refer_to_specialist']}")
    print(f"  top pred = {o['top_class_name']}  prob={o['predictions'][0]['probability']:.2f}  "
          f"entropy={o['entropy']:.2f}")

# ---- (2) deployed DermaMNIST-TEST through full path (fast: replicate routing, no heatmap) ----
m = res["model"]; dev = res["device"]
d = np.load(os.path.join(inf.PROJECT_ROOT, "data", "raw", "dermamnist.npz"))
X, Y = d["test_images"], d["test_labels"].ravel()
probe = res["modality_probe"]; sm = res["serving_multi"]; pm = res["prototypes_multi"]
# band-28 dermoscopy protos
P28 = torch.stack([(pm[28][c] / pm[28][c].norm()) for c in range(7)]).to(dev)
temp = sm["shared"]["temp_metric"]
# clinical protos (5 classes) + mask
clin = res["clinical_protos"]; clin_av = {int(c) for c in clin.keys()}
Pc = torch.zeros(7, P28.shape[1]).to(dev)
for c in clin_av: Pc[c] = clin[c] / clin[c].norm()
mask = np.array([0.0 if c in clin_av else -np.inf for c in range(7)])

def enc(arrs, bs=128):
    out = []
    with torch.no_grad():
        for i in range(0, len(arrs), bs):
            t = torch.stack([m.preprocess(Image.fromarray(a)) for a in arrs[i:i+bs]]).to(dev)
            e = m.backbone.encode_image(t); e = e / e.norm(dim=-1, keepdim=True)
            out.append(e.cpu())
    return torch.cat(out)

E = enc(X)
pclin = 1 / (1 + np.exp(-(E.numpy() @ probe["coef"][0] + probe["intercept"])))
route_clin = pclin > probe["T"]
zc = (E.to(dev) @ Pc.T).cpu().numpy() + mask           # clinical cosine logits (masked)
zd = (-torch.cdist(E.to(dev), P28) * temp).cpu().numpy()  # dermoscopy band-28 logits
pred = np.where(route_clin, zc.argmax(1), zd.argmax(1))
acc = (pred == Y).mean() * 100
print(f"\nDermaMNIST-TEST deployed FULL path (n={len(Y)}): acc={acc:.2f}%  "
      f"routed-clinical={route_clin.mean()*100:.1f}%")
print(f"  (band-28-only, no modality routing, was 38.80%)")
