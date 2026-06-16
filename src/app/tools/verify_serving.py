"""Verify the integrated serving path: metadata fusion + calibration + open-set
rejection + clinical-note gating, on in-distribution HAM10000 images and the
far-OOD clinical webp."""
import os, sys
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, root)
from PIL import Image
from src.app.backend.inference import load_resources, predict
from src.data.preprocessing import load_ham10000

R = load_resources()
sv = R["serving"]
print("serving loaded:", sv is not None,
      "| alpha", sv["meta_alpha"], "| calib_T", sv["calib_T"],
      "| ood", sv["ood_method"], "tau", sv["ood_tau"], "ece_after", sv["ece_after_test"])

paths, labels = load_ham10000(data_root=os.path.join(root, "data", "ham10000"),
                              split="test", val_size=0.1, test_size=0.1, seed=42)
df = [p for p, l in zip(paths, labels) if int(l) == 3][:3]   # dermatofibroma
ak = [p for p, l in zip(paths, labels) if int(l) == 0][:3]   # actinic keratoses

def show(tag, img, meta=None, note=None):
    out = predict(img, R, metadata=meta, clinical_note=note)
    top = ", ".join(f"{p['class_name'][:14]}={p['probability']:.2f}" for p in out["predictions"])
    print(f"  {tag}")
    print(f"    top3: {top}")
    print(f"    unknown={out['is_unknown']} ood={out['ood_score']:.2f} ece={out['calibration_ece']} "
          f"meta_used={out['metadata_used']} meta_changed={out['metadata_changed_ranking']} "
          f"note_used={out['note_used']} note_supports={out['note_supports']}")

print("\nIn-distribution dermatofibroma (no meta) vs (age 35, female, lower extremity):")
img = Image.open(df[0]).convert("RGB")
show("df[0] image-only", img)
show("df[0] + metadata", img, meta={"age": 35, "sex": "F", "localization": "lower extremity"})

print("\nActinic keratoses + epidemiology-consistent metadata (age 75, male, face):")
show("ak[0] + metadata", Image.open(ak[0]).convert("RGB"),
     meta={"age": 75, "sex": "M", "localization": "face"})

print("\nClinical-note gating (generic vs discriminative):")
show("ak[0] generic note", Image.open(ak[0]).convert("RGB"), note="lesion on skin")
show("ak[0] discriminative note", Image.open(ak[0]).convert("RGB"),
     note="rough scaly hyperkeratotic patch on chronically sun-damaged skin in an elderly patient")

print("\nFAR-OOD clinical wide-field webp (should be flagged unknown):")
webp = os.path.join(root, "dermafibroma-sample.webp")
if os.path.exists(webp):
    show("dermafibroma-sample.webp", Image.open(webp).convert("RGB"))
else:
    print("  (webp not found)")
