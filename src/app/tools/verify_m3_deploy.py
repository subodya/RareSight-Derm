"""Verify the deployed M3 inference path: loads new assets, runs predict() on the
OOD clinical webp and on in-distribution HAM10000 dermoscopy dermatofibromas."""
import os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)
from PIL import Image
from src.app.backend.inference import load_resources, predict
from src.data.preprocessing import load_ham10000

R = load_resources()
print("blend_params loaded:", R["blend_params"] is not None,
      "| beta", R["blend_params"]["beta"], "lam", R["blend_params"]["lam"])

def show(tag, img):
    out = predict(img, R)
    top = ", ".join(f"{p['class_name']}={p['probability']:.2f}" for p in out["predictions"])
    print(f"  {tag:<34} -> {top}")

print("\nOOD clinical photo (domain shift expected):")
show("dermafibroma-sample.webp", Image.open(os.path.join(project_root, "dermafibroma-sample.webp")).convert("RGB"))

print("\nIn-distribution HAM10000 dermoscopy Dermatofibroma (class 3) test images:")
paths, labels = load_ham10000(data_root=os.path.join(project_root, "data", "ham10000"),
                              split="test", val_size=0.1, test_size=0.1, seed=42)
df = [p for p, l in zip(paths, labels) if int(l) == 3][:6]
correct = 0
for i, p in enumerate(df):
    out = predict(Image.open(p).convert("RGB"), R)
    pred = out["predictions"][0]["class_name"]
    ok = "OK " if out["top_class_id"] == 3 else "   "
    if out["top_class_id"] == 3: correct += 1
    print(f"  {ok}[{i}] top={pred:<22} p={out['predictions'][0]['probability']:.2f}")
print(f"\n  Dermatofibroma top-1 correct: {correct}/{len(df)}")
