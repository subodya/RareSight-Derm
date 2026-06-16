"""Verify metadata fusion now fires on the CLINICAL path, on the 5 demo clinical samples.
Runs the deployed predict() with vs without patient metadata (note disabled to isolate the
metadata effect) and shows metadata_used + the prediction/confidence change."""
import sys, os, json
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.app.backend.inference import load_resources, predict

man = json.load(open(os.path.join(ROOT, "demo_test_samples", "manifest.json")))
clinical = [s for s in man if s["track"] == "clinical_pad"]

print("Loading resources (fresh — picks up patched clinical_serving_params.pt)...")
res = load_resources()
cs = res.get("clinical_serving") or {}
print(f"clinical_serving has meta_logtab: {'meta_logtab' in cs}  "
      f"meta_alpha={cs.get('meta_alpha')}  calib_T_meta={cs.get('calib_T_meta')}\n")

def top(img, meta, note):
    r = predict(img, res, metadata=meta, clinical_note=note)
    p = r["predictions"][0]
    return f"{p['class_name'][:16]:<17}{p['probability']:.2f}", r["metadata_used"]

print(f"{'sample':<22}{'true':<7}{'image-only':<24}{'+meta':<24}{'+note':<24}{'+note+meta (APP)':<24}")
print("-" * 122)
ok_app = 0
for s in clinical:
    path = os.path.join(ROOT, s["file"]); ui = s["ui"]
    img = Image.open(path).convert("RGB")
    meta = {"age": str(ui["age"]) if ui["age"] is not None else None,
            "sex": ui["sex"], "localization": ui["anatomical_site"]}
    note = ui["clinical_note"]
    c_img, _ = top(img, None, None)
    c_meta, mu = top(img, meta, None)
    c_note, _ = top(img, None, note)
    c_app, _ = top(img, meta, note)
    app_pred = predict(img, res, metadata=meta, clinical_note=note)
    correct = app_pred["top_class_id"] == {"akiec":0,"bcc":1,"bkl":2,"mel":4,"nv":5}[s["true_class"]]
    ok_app += correct
    print(f"{os.path.basename(s['file']):<22}{s['true_class']:<7}{c_img:<24}{c_meta:<24}{c_note:<24}"
          f"{c_app:<20}{'OK' if correct else 'XX'}")
print(f"\nmeta_used fired on clinical: {mu}")
print(f"Realistic app (note+metadata) correct: {ok_app}/{len(clinical)}")
