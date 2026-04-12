"""
Analyze label mask colors to identify corrosion vs spalling classes
"""
from PIL import Image
import numpy as np
from pathlib import Path
import collections

TRAIN_DIR = Path("data/extra/spalling_corrosion_patches/train")

# Check first 10 label files
label_files = list(TRAIN_DIR.glob("*_lab.png"))[:10]
color_counts = collections.Counter()

for lf in label_files:
    img = np.array(Image.open(lf))
    if len(img.shape) == 3:
        pixels = img.reshape(-1, img.shape[2])
        unique = np.unique(pixels, axis=0)
        for u in unique:
            color_counts[tuple(u)] += 1
    else:
        unique = np.unique(img)
        for u in unique:
            color_counts[(int(u),)] += 1

print("Unique colors found in label masks:")
for color, count in color_counts.most_common():
    print(f"  {color} → {count} files")