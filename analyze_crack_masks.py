from PIL import Image
import numpy as np
from pathlib import Path

MASK_DIR = Path("data/crack_seg/UAV-based crack dataset used for segmentation/masks")
masks = list(MASK_DIR.glob("*.png"))[:10]

for m in masks:
    img = np.array(Image.open(m).convert("RGB"))
    unique = np.unique(img.reshape(-1, img.shape[2]), axis=0)
    print(f"{m.name}: {unique[:5]}")
