"""
Does metadata still STACK on CoOp in the deployed regime, or did CoOp's learned text
already absorb the age/sex/site signal? Reports test macro-F1 at the deployed operating
point (7-way, alpha=0.25) for M3 and CoOp, with and without metadata fusion — the actual
number the app produces (the recommended path applies metadata).

Reuses build_serving_artifacts' exact metadata machinery. M3 prototypes from the backup
(pre-deploy), CoOp prototypes = the live deployed ones.

Run:  python src/training/eval_deployed_metadata_stack.py
"""
import sys, os, json, torch, numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.models.raresight_net import RareSight
from src.app.tools.build_serving_artifacts import (
    load_split_meta, fit_meta_logtab, meta_loglik, encode, macroF1)
from sklearn.metrics import accuracy_score

CKPT = "checkpoints/raresight_nblk4mix.pth"
P_M3   = "src/app/assets/_m3_deploy_backup_20260609/disease_prototypes.pt"
P_COOP = "src/app/assets/disease_prototypes.pt"   # live (deployed CoOp)
ALPHA = 0.25                                        # deployed metadata weight
OUT = "checkpoints/deployed_metadata_stack.json"


def _norm(x): return x / x.norm(dim=-1, keepdim=True)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = RareSight(device=dev); m.load_state_dict(torch.load(CKPT, map_location=dev), strict=False); m.eval()
    temp = m.temperature.item()

    def protos(path):
        d = torch.load(path, map_location=dev)
        return torch.stack([_norm(d[c]) for c in range(7)]).to(dev)
    P_m3, P_coop = protos(P_M3), protos(P_COOP)

    sp = load_split_meta("data/ham10000", seed=42)
    logtab, si, xi = fit_meta_logtab(sp["train"])
    y = sp["test"]["label"].values
    Q = encode(m, dev, sp["test"]["path"].tolist()).to(dev)
    mll = meta_loglik(sp["test"], logtab, si, xi)            # (n,7)

    def z(P): return (-torch.cdist(Q, P) * temp).cpu().numpy()
    z_m3, z_coop = z(P_m3), z(P_coop)

    def row(name, zz):
        f0 = 100 * macroF1(y, zz.argmax(1))
        a0 = 100 * accuracy_score(y, zz.argmax(1))
        fm = 100 * macroF1(y, (zz + ALPHA * mll).argmax(1))
        am = 100 * accuracy_score(y, (zz + ALPHA * mll).argmax(1))
        print(f"  {name:<6} no-meta: acc={a0:.2f} mF1={f0:.2f}   +meta(a={ALPHA}): acc={am:.2f} mF1={fm:.2f}   "
              f"meta-stack mF1 {fm-f0:+.2f}")
        return {"acc": round(a0,2), "macro_f1": round(f0,2),
                "acc_meta": round(am,2), "macro_f1_meta": round(fm,2),
                "meta_stack_mf1": round(fm-f0,2)}

    print(f"\n=== DEPLOYED 7-way test, alpha={ALPHA} (metadata stacking) ===")
    res = {"protocol": "7way_hamtest_deployed", "alpha": ALPHA,
           "M3": row("M3", z_m3), "CoOp": row("CoOp", z_coop)}
    res["coop_meta_vs_m3_meta_mf1"] = round(res["CoOp"]["macro_f1_meta"] - res["M3"]["macro_f1_meta"], 2)
    print(f"\n  CoOp+meta vs M3+meta (apples-to-apples deployed):  macro-F1 {res['coop_meta_vs_m3_meta_mf1']:+.2f}")
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
