"""
Extract corrosion and spalling images from segmentation dataset
into classification folders for training
"""
from PIL import Image
import numpy as np
import shutil
from pathlib import Path

TRAIN_DIR = Path("data/extra/spalling_corrosion_patches/train")
VAL_DIR   = Path("data/extra/spalling_corrosion_patches/val")
DEST      = Path("data/codebrim")

CORROSION_COLOR = (255, 0, 0)
SPALLING_COLOR  = (255, 255, 0)
THRESHOLD       = 100  # min pixels of class to count as that class

def classify_image(label_path):
    img = np.array(Image.open(label_path).convert("RGB"))
    corrosion = np.sum(np.all(img == CORROSION_COLOR, axis=2))
    spalling  = np.sum(np.all(img == SPALLING_COLOR,  axis=2))
    if corrosion >= THRESHOLD and corrosion >= spalling:
        return "corrosion"
    elif spalling >= THRESHOLD:
        return "spalling"
    return None

counts = {"corrosion": 0, "spalling": 0, "skipped": 0}

for phase, src_dir in [("train", TRAIN_DIR), ("val", VAL_DIR)]:
    label_files = list(src_dir.glob("*_lab.png"))
    for lf in label_files:
        cls = classify_image(lf)
        if cls is None:
            counts["skipped"] += 1
            continue
        img_file = Path(str(lf).replace("_lab.png", ".png"))
        if not img_file.exists():
            continue
        out_dir = DEST / phase / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(img_file, out_dir / img_file.name)
        counts[cls] += 1

print(f"Corrosion images copied : {counts['corrosion']}")
print(f"Spalling images copied  : {counts['spalling']}")
print(f"Skipped (no class)      : {counts['skipped']}")

# Final count
print("\nFinal dataset structure:")
for phase in ["train", "val"]:
    for cls in ["crack", "intact", "corrosion", "spalling"]:
        p = DEST / phase / cls
        count = len(list(p.glob("*"))) if p.exists() else 0
        print(f"  {phase}/{cls}: {count} images")