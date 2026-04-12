"""
Augment spalling class to fix class imbalance
Author: Seyed Farhad Abtahi
"""
from PIL import Image, ImageEnhance, ImageFilter
import os, random, shutil
from pathlib import Path

random.seed(42)

SRC_TRAIN = Path("data/codebrim/train/spalling")
SRC_VAL   = Path("data/codebrim/val/spalling")
TARGET_TRAIN = 900
TARGET_VAL   = 200

def augment_image(img):
    ops = [
        lambda x: x.transpose(Image.FLIP_LEFT_RIGHT),
        lambda x: x.transpose(Image.FLIP_TOP_BOTTOM),
        lambda x: x.rotate(random.choice([90, 180, 270])),
        lambda x: x.rotate(random.uniform(-20, 20)),
        lambda x: ImageEnhance.Brightness(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: ImageEnhance.Contrast(x).enhance(random.uniform(0.7, 1.3)),
        lambda x: ImageEnhance.Color(x).enhance(random.uniform(0.8, 1.2)),
        lambda x: x.filter(ImageFilter.GaussianBlur(radius=random.uniform(0, 1.5))),
    ]
    chosen = random.sample(ops, random.randint(2, 4))
    for op in chosen:
        img = op(img)
    return img

def augment_folder(src, target_count):
    images = list(src.glob("*.jpg")) + list(src.glob("*.png")) + list(src.glob("*.jpeg"))
    current = len(images)
    needed  = target_count - current
    print(f"  {src.name}: {current} → target {target_count} (adding {needed})")

    counter = 0
    while counter < needed:
        src_img = random.choice(images)
        img     = Image.open(src_img).convert("RGB")
        aug     = augment_image(img)
        out_name = src / f"aug_{counter:04d}_{src_img.stem}.jpg"
        aug.save(out_name, quality=90)
        counter += 1

    print(f"  Done: {len(list(src.glob('*')))} total images")

print("Augmenting train/spalling...")
augment_folder(SRC_TRAIN, TARGET_TRAIN)

print("Augmenting val/spalling...")
augment_folder(SRC_VAL, TARGET_VAL)

print("\nFinal counts:")
for phase in ["train", "val"]:
    for cls in ["crack", "intact", "corrosion", "spalling"]:
        p = Path("data/codebrim") / phase / cls
        print(f"  {phase}/{cls}: {len(list(p.glob('*')))}")