"""
Stage 3: UAV Drone Simulation for Structural Damage Inspection
Simulates a drone flying a grid path, running inference on each frame,
and outputting a damage map + inspection report.
Author: Seyed Farhad Abtahi
"""

import os
import json
import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH  = "models/best_resnet50_multiclass.pth"
DATA_DIR    = "data/codebrim"
RESULTS_DIR = "results"
GRID_ROWS   = 4
GRID_COLS   = 5
NUM_CLASSES = 4
CLASSES     = ["crack", "intact", "corrosion", "spalling"]
DEVICE      = torch.device("cpu")

CLASS_COLORS = {
    "crack":     "#FF4444",
    "corrosion": "#FF8800",
    "spalling":  "#FFCC00",
    "intact":    "#44BB44"
}

SEVERITY = {
    "crack":     "CRITICAL",
    "corrosion": "HIGH",
    "spalling":  "MEDIUM",
    "intact":    "NONE"
}

print("Loading model...")

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()
print("Model loaded.")

# ─────────────────────────────────────────────
# IMAGE TRANSFORM
# ─────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────────
# COLLECT SAMPLE IMAGES
# ─────────────────────────────────────────────
def collect_images(data_dir, n=20):
    images = []
    for cls in os.listdir(os.path.join(data_dir, "val")):
        cls_dir = os.path.join(data_dir, "val", cls)
        if not os.path.isdir(cls_dir):
            continue
        files = [os.path.join(cls_dir, f) for f in os.listdir(cls_dir)
                 if f.endswith((".jpg", ".png", ".jpeg"))]
        images.extend(files[:n//2])
    random.shuffle(images)
    return images[:GRID_ROWS * GRID_COLS]

# ─────────────────────────────────────────────
# RUN INFERENCE
# ─────────────────────────────────────────────
def predict(img_path):
    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)
        conf, pred = probs.max(dim=1)
    return CLASSES[pred.item()], round(conf.item() * 100, 1), img

# ─────────────────────────────────────────────
# SIMULATE DRONE FLIGHT
# ─────────────────────────────────────────────
print("Simulating drone flight...")
images    = collect_images(DATA_DIR)
grid      = []
path      = []
report    = []

for row in range(GRID_ROWS):
    grid_row = []
    cols     = range(GRID_COLS) if row % 2 == 0 else range(GRID_COLS-1, -1, -1)
    for col in cols:
        idx = row * GRID_COLS + col
        if idx >= len(images):
            continue
        label, confidence, img = predict(images[idx])
        cell = {
            "row": row, "col": col,
            "label": label,
            "confidence": confidence,
            "path": images[idx]
        }
        grid_row.append(cell)
        path.append((row, col))
        report.append(cell)
        print(f"  Frame ({row},{col}): {label} {confidence}%")
    grid.append(grid_row)

# ─────────────────────────────────────────────
# DAMAGE MAP
# ─────────────────────────────────────────────
print("\nGenerating damage map...")
fig, axes = plt.subplots(GRID_ROWS, GRID_COLS,
                         figsize=(GRID_COLS*3, GRID_ROWS*3))
fig.patch.set_facecolor("#1a1a2e")

for row in range(GRID_ROWS):
    for col in range(GRID_COLS):
        ax  = axes[row][col]
        idx = row * GRID_COLS + col
        ax.set_facecolor("#1a1a2e")
        ax.set_xticks([]); ax.set_yticks([])

        if idx >= len(images):
            continue

        cell  = next((c for c in report if c["row"]==row and c["col"]==col), None)
        if cell is None:
            continue

        img   = Image.open(cell["path"]).convert("RGB")
        color = CLASS_COLORS.get(cell["label"], "#888888")

        ax.imshow(img)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)

        ax.set_title(f"{cell['label'].upper()}\n{cell['confidence']}%",
                     color=color, fontsize=8, fontweight="bold", pad=3)

        sev = SEVERITY.get(cell["label"], "")
        ax.text(0.02, 0.02, sev, transform=ax.transAxes,
                color=color, fontsize=6, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          facecolor="#1a1a2e", alpha=0.8))

patches = [mpatches.Patch(color=v, label=k.upper())
           for k, v in CLASS_COLORS.items()]
fig.legend(handles=patches, loc="lower center", ncol=4,
           facecolor="#1a1a2e", labelcolor="white", fontsize=10)

plt.suptitle("UAV STRUCTURAL INSPECTION — DAMAGE MAP",
             color="white", fontsize=14, fontweight="bold",
             y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "drone_damage_map.png"),
            dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
plt.show()

# ─────────────────────────────────────────────
# INSPECTION REPORT
# ─────────────────────────────────────────────
counts = {}
for cell in report:
    counts[cell["label"]] = counts.get(cell["label"], 0) + 1

total   = len(report)
damaged = total - counts.get("intact", 0)

print("\n" + "="*50)
print("INSPECTION REPORT")
print("="*50)
print(f"Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"Grid       : {GRID_ROWS}x{GRID_COLS} ({total} frames)")
print(f"Damaged    : {damaged}/{total} zones ({round(damaged/total*100,1)}%)")
for cls, count in counts.items():
    print(f"  {cls:12s}: {count} zones")
print("="*50)

report_data = {
    "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
    "grid": f"{GRID_ROWS}x{GRID_COLS}",
    "total_frames": total,
    "damaged_zones": damaged,
    "damage_rate": round(damaged/total*100, 1),
    "class_counts": counts,
    "frames": report
}

with open(os.path.join(RESULTS_DIR, "metrics", "drone_report.json"), "w") as f:
    json.dump(report_data, f, indent=2)

print("\nSaved: results/plots/drone_damage_map.png")
print("Saved: results/metrics/drone_report.json")
print("Stage 3 Complete.")