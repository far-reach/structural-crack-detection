"""
Dataset Setup for SDNET2018 → 4-class format
Maps: Cracked → crack, Uncracked → intact
Corrosion + Spalling = small synthetic folders (created empty, filled later)
Author: Seyed Farhad Abtahi
"""

import os, shutil, random
from pathlib import Path

CODEBRIM  = Path("data/codebrim")
DEST      = Path("data/codebrim")
SPLIT     = 0.8
random.seed(42)

# ── Collect all cracked / uncracked images from all 3 structure types
cracked_imgs   = []
uncracked_imgs = []

for structure in ["Decks", "Walls", "Pavements"]:
    for sub in (CODEBRIM / structure).iterdir():
        imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png")) + list(sub.glob("*.jpeg"))
        if sub.name.lower().startswith("n"):  # Non-cracked
            uncracked_imgs.extend(imgs)
        else:                                  # Cracked
            cracked_imgs.extend(imgs)

print(f"Cracked images found   : {len(cracked_imgs)}")
print(f"Uncracked images found : {len(uncracked_imgs)}")

# ── Balance: cap intact at same count as cracked to avoid imbalance
random.shuffle(cracked_imgs)
random.shuffle(uncracked_imgs)
min_count      = min(len(cracked_imgs), len(uncracked_imgs), 3000)
cracked_imgs   = cracked_imgs[:min_count]
uncracked_imgs = uncracked_imgs[:min_count]

print(f"Using {min_count} images per class (balanced)")

# ── Split and copy crack / intact
for cls, imgs in [("crack", cracked_imgs), ("intact", uncracked_imgs)]:
    random.shuffle(imgs)
    split = int(len(imgs) * SPLIT)
    for phase, files in [("train", imgs[:split]), ("val", imgs[split:])]:
        out = DEST / phase / cls
        out.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, out / f.name)
    print(f"  {cls:10s}: {split} train / {len(imgs)-split} val")

# ── Create placeholder folders for corrosion + spalling
# (We will add real images in Stage 1b or use augmentation)
for cls in ["corrosion", "spalling"]:
    for phase in ["train", "val"]:
        (DEST / phase / cls).mkdir(parents=True, exist_ok=True)
    print(f"  {cls:10s}: placeholder folders created (add images manually)")

print("\n✅ Dataset setup complete.")
print("\nFinal structure:")
for phase in ["train", "val"]:
    for cls in ["crack", "intact", "corrosion", "spalling"]:
        p = DEST / phase / cls
        count = len(list(p.glob("*"))) if p.exists() else 0
        print(f"  {phase}/{cls}: {count} images")