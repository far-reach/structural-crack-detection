"""Re-run Stage 1 evaluation and the Stage 3 grid on the rebuilt val split.

Uses the released v1.0 weights and the class order that training actually
used (alphabetical ImageFolder order). Writes results to
results_v2/audit/s3_audit.json. Read-only for the repo.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from inspectswarm.classes import V1_CLASSES, V1_WRONG_DECODE_ORDER
from inspectswarm.paths import ROOT as _ROOT, REPO as _REPO, RESULTS as _RESULTS

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import datasets, models, transforms

ROOT = _ROOT
REPO = _REPO
VAL = ROOT / "rebuild/codebrim/val"
OUT = _RESULTS / "audit"
OUT.mkdir(parents=True, exist_ok=True)
WRONG = list(V1_WRONG_DECODE_ORDER)  # hardcoded list in the four scripts

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
torch.manual_seed(42)
m = models.resnet50(weights=None)
m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 256), nn.ReLU(),
                     nn.Dropout(0.3), nn.Linear(256, 4))
m.load_state_dict(torch.load(REPO / "models/best_resnet50_multiclass.pth", map_location=dev))
m.eval().to(dev)
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

ds = datasets.ImageFolder(str(VAL), tf)
CLASSES = ds.classes
assert tuple(CLASSES) == V1_CLASSES, CLASSES
loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
y, p, conf = [], [], []
with torch.no_grad():
    for x, t in loader:
        pr = torch.softmax(m(x.to(dev)), 1).cpu()
        y += t.tolist(); p += pr.argmax(1).tolist(); conf += pr.max(1).values.tolist()
y, p, conf = np.array(y), np.array(p), np.array(conf)
names = [Path(s[0]).name for s in ds.samples]
is_aug = np.array([n.startswith("aug_") for n in names])

res = {"device": str(dev), "classes": CLASSES}
res["all_1600"] = {"acc": float((y == p).mean()), "cm": confusion_matrix(y, p).tolist()}
res["originals_only"] = {"n": int((~is_aug).sum()), "acc": float((y == p)[~is_aug].mean()),
                         "cm": confusion_matrix(y[~is_aug], p[~is_aug], labels=[0, 1, 2, 3]).tolist()}
res["augmented_only"] = {"n": int(is_aug.sum()), "acc": float((y == p)[is_aug].mean())}
print("ALL 1600: acc", res["all_1600"]["acc"]); print(np.array(res["all_1600"]["cm"]))
print(classification_report(y, p, target_names=CLASSES, digits=4))
print("ORIGINALS ONLY n =", res["originals_only"]["n"], "acc", res["originals_only"]["acc"])
print(np.array(res["originals_only"]["cm"]))
print(classification_report(y[~is_aug], p[~is_aug], target_names=CLASSES, digits=4))

# Stage 3: the 20 frames of the published report, in published order
rep = json.load(open(REPO / "results/metrics/drone_report.json"))
idx = {(Path(s[0]).parent.name, Path(s[0]).name): i for i, s in enumerate(ds.samples)}
frames = []
for fr in rep["frames"]:
    parts = fr["path"].replace("\\", "/").split("/")
    i = idx[(parts[-2], parts[-1])]
    frames.append({"row": fr["row"], "col": fr["col"], "file": f"{parts[-2]}/{parts[-1]}",
                   "true": parts[-2], "published_label": fr["label"],
                   "published_conf": fr["confidence"], "pred_index": int(p[i]),
                   "correct_label": CLASSES[p[i]], "wrong_list_label": WRONG[p[i]],
                   "conf": round(float(conf[i]) * 100, 1)})
res["stage3_frames"] = frames
repro = sum(f["wrong_list_label"] == f["published_label"] for f in frames)
true_dmg = [f for f in frames if f["true"] != "intact"]
true_int = [f for f in frames if f["true"] == "intact"]
s3 = {
    "frames_where_wrong_list_reproduces_published_label": repro,
    "max_abs_conf_diff_vs_published": max(abs(f["conf"] - f["published_conf"]) for f in frames),
    "true_damaged": len(true_dmg), "true_intact": len(true_int),
    "damaged_flagged_as_damaged": sum(f["correct_label"] != "intact" for f in true_dmg),
    "damaged_with_correct_class": sum(f["correct_label"] == f["true"] for f in true_dmg),
    "intact_false_alarms": sum(f["correct_label"] != "intact" for f in true_int),
    "exact_class_correct": sum(f["correct_label"] == f["true"] for f in frames),
    "published_exact_class_correct": sum(f["published_label"] == f["true"] for f in frames),
    "distinct_parent_images": {c: sorted({f["file"].split("/")[1].split("_")[0].split("-")[0]
                                          for f in frames if f["true"] == c})
                               for c in CLASSES},
}
res["stage3_summary"] = s3
print("STAGE 3:", json.dumps(s3, indent=1))
for f in frames:
    print(f"  r{f['row']}c{f['col']} {f['file']:32s} true={f['true']:9s} published={f['published_label']:9s} "
          f"({f['published_conf']}) corrected={f['correct_label']:9s} ({f['conf']})")
json.dump(res, open(OUT / "s3_audit.json", "w"), indent=1)
