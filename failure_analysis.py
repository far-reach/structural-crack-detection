"""
Failure Analysis Figure - Fixed Version
- No embedded title (goes in caption below figure per journal style)
- Finds 3 DIFFERENT failure modes, not just one
- More top padding to prevent label clipping
Author: Seyed Farhad Abtahi
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os

DEVICE     = torch.device("cpu")
MODEL_PATH = "models/best_resnet50_multiclass.pth"
DATA_DIR   = "data/codebrim/val"
CLASSES    = ["crack", "intact", "corrosion", "spalling"]

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(model.fc.in_features, 256),
    nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def predict(img_path):
    img    = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs  = torch.softmax(output, dim=1)[0]
        pred   = probs.argmax().item()
        conf   = probs[pred].item()
    return CLASSES[pred], round(conf*100, 1), img

# Find ONE failure case per true class (diversity)
failures = []
seen_true = set()
seen_false = set()

for true_cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, true_cls)
    if not os.path.exists(cls_dir):
        continue
    files = [f for f in os.listdir(cls_dir)
             if f.endswith(('.jpg','.png','.jpeg'))]
    for fname in files:
        path = os.path.join(cls_dir, fname)
        pred_cls, conf, img = predict(path)
        if pred_cls != true_cls and true_cls not in seen_true:
            failures.append({
                "path": path, "true": true_cls,
                "pred": pred_cls, "conf": conf, "img": img
            })
            seen_true.add(true_cls)
            break
    if len(failures) >= 3:
        break

print(f"Found {len(failures)} distinct failure cases:")
for f in failures:
    print(f"  True: {f['true']:12s} Pred: {f['pred']:12s} Conf: {f['conf']}%")

# Failure descriptions for annotation
descriptions = {
    ("crack", "intact"):     "Hairline crack\nlow contrast\n→ missed",
    ("crack", "corrosion"):  "Crack near rust\nstaining\n→ confused",
    ("crack", "spalling"):   "Crack with\nflaking surface\n→ confused",
    ("corrosion", "intact"): "Early corrosion\nlow saturation\n→ missed",
    ("corrosion", "crack"):  "Rust streak\nlinear pattern\n→ confused",
    ("corrosion", "spalling"): "Corroded edge\nwith loss\n→ confused",
    ("spalling", "intact"):  "Fine spalling\nnear edge\n→ missed",
    ("spalling", "crack"):   "Spalling with\nfine cracks\n→ confused",
    ("intact", "crack"):     "Texture pattern\nresembles crack\n→ false positive",
    ("intact", "corrosion"): "Reddish tint\nresembles rust\n→ false positive",
}

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.subplots_adjust(top=0.82, bottom=0.05, left=0.04, right=0.96, wspace=0.15)

for i, f in enumerate(failures[:3]):
    axes[i].imshow(f["img"].resize((224, 224)))
    key = (f["true"], f["pred"])
    desc = descriptions.get(key, "Misclassification")
    axes[i].set_title(
        f"True: {f['true'].upper()}\n"
        f"Pred: {f['pred'].upper()} ({f['conf']}%)\n"
        f"{desc}",
        fontsize=10, color="#AA2200", pad=8,
        linespacing=1.4
    )
    axes[i].axis("off")
    # Add border
    for spine in axes[i].spines.values():
        spine.set_visible(True)
        spine.set_edgecolor('#AA2200')
        spine.set_linewidth(2)

plt.savefig("results/plots/failure_analysis.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: results/plots/failure_analysis.png")