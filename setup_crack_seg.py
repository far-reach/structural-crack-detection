"""
Split UAV crack segmentation dataset into train/val
Author: Seyed Farhad Abtahi
"""
import os, shutil, random
from pathlib import Path

SRC_IMG  = Path("data/crack_seg/UAV-based crack dataset used for segmentation/image")
SRC_MASK = Path("data/crack_seg/UAV-based crack dataset used for segmentation/masks")
DEST     = Path("data/crack_seg")
SPLIT    = 0.8
random.seed(42)

images = sorted(list(SRC_IMG.glob("*.png")))
random.shuffle(images)
split  = int(len(images) * SPLIT)

for phase, files in [("train", images[:split]), ("val", images[split:])]:
    (DEST / phase / "images").mkdir(parents=True, exist_ok=True)
    (DEST / phase / "masks").mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy(f, DEST / phase / "images" / f.name)
        mask = SRC_MASK / f.name
        if mask.exists():
            shutil.copy(mask, DEST / phase / "masks" / f.name)

print(f"Train: {split} | Val: {len(images)-split}")
print("Done.")