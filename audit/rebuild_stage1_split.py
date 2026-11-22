"""Rebuild the Stage 1 train/val folders of the v3 preprint.

Mirrors, step by step, the logic and random-call sequence of the backbone
scripts setup_dataset.py, extract_classes.py, augment_corrosion.py and
augment_spalling.py, with only the paths changed. Nothing in the repo clone
is touched; output goes to <INSPECTSWARM_ROOT>/rebuild/codebrim.

Usage: python rebuild_stage1_split.py [--no-aug]
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from inspectswarm.classes import V1_CLASSES, V1_WRONG_DECODE_ORDER
from inspectswarm.paths import ROOT as _ROOT, REPO as _REPO, RESULTS as _RESULTS

import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = _ROOT
SDNET = ROOT / "datasets/sdnet2018"
RAID = ROOT / "datasets/raidathmane/spalling_corrosion_patches"
DEST = ROOT / "rebuild/codebrim"
ORDER = list(V1_CLASSES)


def count(phase, cls):
    p = DEST / phase / cls
    return len(list(p.glob("*"))) if p.exists() else 0


def step1_sdnet():
    """setup_dataset.py"""
    random.seed(42)
    cracked, uncracked = [], []
    for structure in ["Decks", "Walls", "Pavements"]:
        for sub in (SDNET / structure).iterdir():
            imgs = list(sub.glob("*.jpg")) + list(sub.glob("*.png")) + list(sub.glob("*.jpeg"))
            if sub.name.lower().startswith("n"):
                uncracked.extend(imgs)
            else:
                cracked.extend(imgs)
    print(f"cracked pool {len(cracked)}, uncracked pool {len(uncracked)}")
    random.shuffle(cracked)
    random.shuffle(uncracked)
    n = min(len(cracked), len(uncracked), 3000)
    cracked, uncracked = cracked[:n], uncracked[:n]
    for cls, imgs in [("crack", cracked), ("intact", uncracked)]:
        random.shuffle(imgs)
        split = int(len(imgs) * 0.8)
        for phase, files in [("train", imgs[:split]), ("val", imgs[split:])]:
            out = DEST / phase / cls
            out.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, out / f.name)


def step2_raidathmane():
    """extract_classes.py"""
    for phase in ["train", "val"]:
        for lf in list((RAID / phase).glob("*_lab.png")):
            a = np.array(Image.open(lf).convert("RGB"))
            corrosion = np.sum(np.all(a == (255, 0, 0), axis=2))
            spalling = np.sum(np.all(a == (255, 255, 0), axis=2))
            if corrosion >= 100 and corrosion >= spalling:
                cls = "corrosion"
            elif spalling >= 100:
                cls = "spalling"
            else:
                continue
            img_file = Path(str(lf).replace("_lab.png", ".png"))
            if not img_file.exists():
                continue
            out = DEST / phase / cls
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy(img_file, out / img_file.name)


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
    for op in random.sample(ops, random.randint(2, 4)):
        img = op(img)
    return img


def augment_folder(src, target):
    images = list(src.glob("*.jpg")) + list(src.glob("*.png")) + list(src.glob("*.jpeg"))
    needed = target - len(images)
    print(f"  {src.parent.name}/{src.name}: {len(images)} originals, adding {needed}")
    for counter in range(max(needed, 0)):
        src_img = random.choice(images)
        aug = augment_image(Image.open(src_img).convert("RGB"))
        aug.save(src / f"aug_{counter:04d}_{src_img.stem}.jpg", quality=90)


def step3_augment(cls):
    """augment_corrosion.py / augment_spalling.py (each script seeds 42 itself)"""
    random.seed(42)
    augment_folder(DEST / "train" / cls, 900)
    augment_folder(DEST / "val" / cls, 200)


if __name__ == "__main__":
    if DEST.exists():
        sys.exit(f"{DEST} already exists; remove it first")
    step1_sdnet()
    step2_raidathmane()
    print("before augmentation:", {f"{p}/{c}": count(p, c) for p in ["train", "val"] for c in ORDER})
    if "--no-aug" not in sys.argv:
        step3_augment("corrosion")
        step3_augment("spalling")
    print("final:", {f"{p}/{c}": count(p, c) for p in ["train", "val"] for c in ORDER})
