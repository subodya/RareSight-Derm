"""
PER-BAND RESOLUTION-MATCHED PROTOTYPES (training-free ceiling for "any resolution").

Fixes the confound from diag_resolution_sweep: ALL prototypes here use the SAME M3/CoOp
text blend as deployed (beta=0.75, lam=1.0, gap). Builds a prototype set per resolution
band {28,56,112,224,450} from TRAIN support downsampled to that band, then for each query
resolution routes to the MATCHED band. Reports:
  - deployed (single full-res blended) protos       [baseline]
  - single robust blended protos (multi-res mix)    [the compromise]
  - per-band matched blended protos                 [the no-tradeoff candidate]
"""
import os, sys
import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.models.raresight_net import RareSight
from src.data.preprocessing import load_ham10000

device = "cuda" if torch.cuda.is_available() else "cpu"
model = RareSight(device=device)
model.load_state_dict(torch.load(os.path.join(ROOT,"checkpoints","raresight_nblk4mix.pth"),
                                 map_location=device), strict=False)
model.eval()
HAM = os.path.join(ROOT, "data", "ham10000")
RES = [28, 56, 112, 224, 450]
rng = np.random.RandomState(42)

blend = torch.load(os.path.join(ROOT,"src","app","assets","blend_params.pt"),
                   map_location="cpu", weights_only=False)
BETA, LAM, GAP = blend["beta"], blend["lam"], blend["gap"]
TXT = blend["text_embs"]

def apply_blend(img_proto, c):
    ts = TXT[c] + LAM * GAP; ts = ts / ts.norm()
    p = BETA * img_proto + (1.0 - BETA) * ts
    return p / p.norm()

def downsample(pil, res):
    if res >= min(pil.size): return pil
    return pil.resize((res, res), Image.BILINEAR)

def encode(pils, bs=64):
    out=[]
    for i in range(0,len(pils),bs):
        t=torch.stack([model.preprocess(p) for p in pils[i:i+bs]]).to(device)
        with torch.no_grad():
            e=model.backbone.encode_image(t); e=e/e.norm(dim=-1,keepdim=True)
        out.append(e.cpu())
    return torch.cat(out)

def mf1(pred,y):
    f=[]
    for c in range(7):
        tp=((pred==c)&(y==c)).sum();fp=((pred==c)&(y!=c)).sum();fn=((pred!=c)&(y==c)).sum()
        p=tp/(tp+fp+1e-9);r=tp/(tp+fn+1e-9);f.append(2*p*r/(p+r+1e-9))
    return np.mean(f)

# test set (40/class)
te_paths, te_y = load_ham10000(HAM, split="test", seed=42); te_y=np.array(te_y)
keep=[]
for c in range(7):
    idx=np.where(te_y==c)[0]; keep+=rng.choice(idx,size=min(40,len(idx)),replace=False).tolist()
keep=np.array(keep); test_full=[Image.open(te_paths[i]).convert("RGB") for i in keep]; test_y=te_y[keep]

# train support indices (fixed K=20/class)
tr_paths, tr_y = load_ham10000(HAM, split="train", seed=42); tr_y=np.array(tr_y)
supp_idx={c: rng.choice(np.where(tr_y==c)[0], size=min(20,(tr_y==c).sum()), replace=False) for c in range(7)}

def build_protos_at(res):
    P=[]
    for c in range(7):
        pils=[downsample(Image.open(tr_paths[i]).convert("RGB"), res) for i in supp_idx[c]]
        e=encode(pils); ip=e.mean(0); ip=ip/ip.norm()
        P.append(apply_blend(ip, c))
    return torch.stack(P)

# per-band protos + single robust (mix) proto
Pband={res: build_protos_at(res) for res in RES}
robust=[]
for c in range(7):
    pils=[downsample(Image.open(tr_paths[i]).convert("RGB"), RES[j%len(RES)]) for j,i in enumerate(supp_idx[c])]
    e=encode(pils); ip=e.mean(0); ip=ip/ip.norm(); robust.append(apply_blend(ip,c))
Probust=torch.stack(robust)

dep=torch.load(os.path.join(ROOT,"src","app","assets","disease_prototypes.pt"),map_location="cpu")
Pdep=torch.stack([dep[c] if c in dep else dep[str(c)] for c in range(7)]).float()
Pdep=Pdep/Pdep.norm(dim=-1,keepdim=True)

print("\nquery |  deployed(blend) | robust-mix(blend) | PER-BAND matched(blend)")
print("  res |  acc    mF1      | acc    mF1        | acc    mF1")
for res in RES:
    E=encode([downsample(p,res) for p in test_full])
    pa=(E@Pdep.T).argmax(1).numpy()
    pr=(E@Probust.T).argmax(1).numpy()
    pb=(E@Pband[res].T).argmax(1).numpy()   # matched band
    print(f"  {res:3d} | {(pa==test_y).mean()*100:5.1f} {mf1(pa,test_y)*100:5.1f}   "
          f"| {(pr==test_y).mean()*100:5.1f} {mf1(pr,test_y)*100:5.1f}    "
          f"| {(pb==test_y).mean()*100:5.1f} {mf1(pb,test_y)*100:5.1f}")
