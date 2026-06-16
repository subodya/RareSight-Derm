"""POST every demo_test_samples image through the REAL /api/predict endpoint (exactly as the
frontend does) and compare to the manifest's expected result. Backend must be running on :8044."""
import json, os, sys, time, urllib.request
import requests

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BASE = "http://127.0.0.1:8044"
manifest = json.load(open(os.path.join(ROOT, "demo_test_samples", "manifest.json")))

# wait for health
for _ in range(120):
    try:
        h = json.load(urllib.request.urlopen(BASE + "/api/health", timeout=2))
        if h.get("status") == "ok":
            print(f"backend ready: device={h['device']} classes={h['classes_loaded']}")
            break
    except Exception:
        time.sleep(1)
else:
    print("backend never came up"); sys.exit(1)

print(f"\n{'image':<40}{'true':<7}{'HTTP top':<22}{'prob':<7}{'mod':<11}{'refer':<6}match")
print("-" * 100)
n_ok = 0
for s in manifest:
    ui = s["ui"]
    path = os.path.join(ROOT, s["file"])
    data = {}
    if ui["age"] is not None:
        data["age"] = str(ui["age"])
    data["sex"] = ui["sex"]
    data["localization"] = ui["anatomical_site"]
    data["scan_type"] = ui["scan_modality"]
    data["clinical_note"] = ui["clinical_note"]
    with open(path, "rb") as f:
        r = requests.post(BASE + "/api/predict", files={"file": f}, data=data, timeout=120)
    r.raise_for_status()
    j = r.json()
    top = j["predictions"][0]
    match = "OK" if top["class_name"].lower().startswith(s["true_class"][:3]) or \
        j["top_class_id"] == {"akiec":0,"bcc":1,"bkl":2,"df":3,"mel":4,"nv":5,"vasc":6}[s["true_class"]] else "XX"
    n_ok += match == "OK"
    print(f"{os.path.basename(s['file']):<40}{s['true_class']:<7}{top['class_name'][:20]:<22}"
          f"{top['probability']:<7.3f}{j['modality']:<11}{str(j['refer_to_specialist']):<6}{match}")
    print(f"{'  expected (manifest):':<40}{'':<7}{s['app_result']['top_name'][:20]:<22}"
          f"{s['app_result']['prob']:<7.3f}{s['app_result']['modality']:<11}")

print(f"\nHTTP path correct: {n_ok}/{len(manifest)}")
