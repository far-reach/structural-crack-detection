"""Audit and regenerate the failure-analysis, Grad-CAM and damage-map figures.

Part A replays the selection logic of failure_analysis.py and fix_gradcam.py
with the wrong hardcoded class list, to show which images the published
figures actually displayed. Part B repeats the same logic with the correct
(alphabetical) order and saves corrected figures to results_v2/audit.
Released v1.0 weights, rebuilt val split, CPU. Read-only for the repo.
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from inspectswarm.classes import V1_CLASSES, V1_WRONG_DECODE_ORDER
from inspectswarm.paths import ROOT as _ROOT, REPO as _REPO, RESULTS as _RESULTS

import json
import os
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = _ROOT
VAL = ROOT / "rebuild/codebrim/val"
OUT = _RESULTS / "audit"
OUT.mkdir(parents=True, exist_ok=True)
RIGHT = list(V1_CLASSES)
WRONG = list(V1_WRONG_DECODE_ORDER)
LOOP = list(V1_WRONG_DECODE_ORDER)  # iteration order used by both scripts
COLORS = {"crack": "#FF4444", "corrosion": "#FF8800", "spalling": "#FFCC00", "intact": "#44BB44"}
SEVERITY = {"crack": "CRITICAL", "corrosion": "HIGH", "spalling": "MEDIUM", "intact": "NONE"}

model = models.resnet50(weights=None)
model.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(model.fc.in_features, 256), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(256, 4))
model.load_state_dict(torch.load(_REPO / "models/best_resnet50_multiclass.pth",
                                 map_location="cpu"))
model.eval()
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
acts, grads = [], []


def _hook(mod, inp, out):
    if out.requires_grad:
        acts.append(out)
        out.register_hook(grads.append)


model.layer4[-1].register_forward_hook(_hook)


def files(cls):
    return [f for f in os.listdir(VAL / cls) if f.endswith((".jpg", ".png", ".jpeg"))]


_cache = {}


def predict(cls, fname):
    if (cls, fname) not in _cache:
        with torch.no_grad():
            pr = torch.softmax(model(tf(Image.open(VAL / cls / fname).convert("RGB")).unsqueeze(0)), 1)[0]
        _cache[(cls, fname)] = (int(pr.argmax()), round(float(pr.max()) * 100, 1))
    return _cache[(cls, fname)]


def gradcam(cls, fname):
    acts.clear(); grads.clear()
    img = Image.open(VAL / cls / fname).convert("RGB")
    out = model(tf(img).unsqueeze(0))
    pred = int(out.argmax(1))
    model.zero_grad()
    out[0, pred].backward()
    g, a = grads[0].squeeze().numpy(), acts[0].squeeze().detach().numpy()
    cam = np.maximum((g.mean(axis=(1, 2))[:, None, None] * a).sum(0), 0)
    cam = cv2.resize(cam, (224, 224)); cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    img_np = np.array(img.resize((224, 224)))
    return img_np, cam, 0.5 * img_np / 255.0 + 0.5 * plt.cm.jet(cam)[:, :, :3]


def pick_failures(names):
    """failure_analysis.py: first file per true class whose decoded label differs; stop at 3."""
    out = []
    for cls in LOOP:
        for f in files(cls):
            k, conf = predict(cls, f)
            if names[k] != cls:
                out.append({"file": f"{cls}/{f}", "true": cls, "shown_pred": names[k],
                            "actual_pred": RIGHT[k], "conf": conf})
                break
        if len(out) >= 3:
            break
    return out


def pick_gradcam(names):
    """fix_gradcam.py: first file per class whose decoded label equals the class, else first file."""
    out = []
    for cls in LOOP:
        chosen, fallback = None, None
        for f in files(cls):
            k, conf = predict(cls, f)
            row = {"file": f"{cls}/{f}", "true": cls, "shown_pred": names[k],
                   "actual_pred": RIGHT[k], "conf": conf}
            if names[k] == cls:
                chosen = row
                break
            fallback = fallback or row
        row = chosen or fallback
        row["shown_as"] = "Correct" if chosen else "Wrong (fallback)"
        row["actually_correct"] = row["actual_pred"] == cls
        out.append(row)
    return out


res = {"failure_published_logic": pick_failures(WRONG), "failure_corrected": pick_failures(RIGHT),
       "gradcam_published_logic": pick_gradcam(WRONG), "gradcam_corrected": pick_gradcam(RIGHT)}
for k, rows in res.items():
    print(k)
    for r in rows:
        print("  ", r)
json.dump(res, open(OUT / "figure_audit.json", "w"), indent=1)

# corrected failure figure
rows = res["failure_corrected"]
fig, axes = plt.subplots(1, len(rows), figsize=(4.7 * len(rows), 5.2))
for ax, r in zip(np.atleast_1d(axes), rows):
    ax.imshow(Image.open(VAL / r["file"]).convert("RGB").resize((224, 224)))
    ax.set_title(f"True: {r['true'].upper()}\nPred: {r['actual_pred'].upper()} ({r['conf']}%)",
                 fontsize=10, color="#AA2200")
    ax.axis("off")
fig.savefig(OUT / "failure_analysis_corrected.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# corrected Grad-CAM figure
rows = res["gradcam_corrected"]
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
for i, r in enumerate(rows):
    cls, f = r["file"].split("/")
    img_np, cam, overlay = gradcam(cls, f)
    axes[i][0].imshow(img_np); axes[i][0].set_title(f"Input: {cls.upper()}", fontsize=11, fontweight="bold")
    axes[i][1].imshow(cam, cmap="jet"); axes[i][1].set_title("Grad-CAM heatmap", fontsize=11)
    axes[i][2].imshow(np.clip(overlay, 0, 1))
    axes[i][2].set_title(f"Pred: {r['actual_pred'].upper()} ({r['conf']}%)", fontsize=11,
                         color="green" if r["actually_correct"] else "red")
    for ax in axes[i]:
        ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "gradcam_corrected.png", dpi=150, bbox_inches="tight", facecolor="white")
plt.close(fig)

# corrected damage map: the 20 published frames, released weights, correct order
rep = json.load(open(_REPO / "results/metrics/drone_report.json"))
fig, axes = plt.subplots(4, 5, figsize=(15, 12)); fig.patch.set_facecolor("#1a1a2e")
for fr in rep["frames"]:
    cls, f = fr["path"].replace("\\", "/").split("/")[-2:]
    k, conf = predict(cls, f)
    lab = RIGHT[k]; ax = axes[fr["row"]][fr["col"]]
    ax.imshow(Image.open(VAL / cls / f).convert("RGB")); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor(COLORS[lab]); s.set_linewidth(4)
    ax.set_title(f"{lab.upper()} {conf}% ({SEVERITY[lab]})\ntrue: {cls}", color=COLORS[lab], fontsize=8,
                 fontweight="bold")
fig.tight_layout()
fig.savefig(OUT / "drone_damage_map_corrected.png", dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
print("figures saved to", OUT)
