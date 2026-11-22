"""Resize CODEBRIM crops once to 256x256 and store them as uint8 arrays.

Output: <cache>/codebrim_<split>.npz with x (N,256,256,3), y (N,5), names.
The official train/val/test split of the dataset is kept unchanged.
"""
import sys
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectswarm.classes import CODEBRIM_DEFECTS
from inspectswarm.paths import CACHE, CODEBRIM

SIZE = 256


def load_labels():
    labels = {}
    for f in ("defects.xml", "background.xml"):
        for d in ET.parse(CODEBRIM / "metadata" / f).getroot():
            name = d.attrib.get("name")
            if name is None:
                continue
            v = {c.tag: int(c.text) for c in d}
            labels[name] = [int(v[k] > 0) for k in CODEBRIM_DEFECTS]
    return labels


def load(path):
    return np.asarray(Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BILINEAR))


if __name__ == "__main__":
    labels = load_labels()
    CACHE.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        files = sorted(p for k in ("defects", "background") for p in (CODEBRIM / split / k).glob("*.png"))
        missing = [p.name for p in files if p.name not in labels]
        assert not missing, missing[:5]
        with ThreadPoolExecutor(16) as ex:
            x = np.stack(list(ex.map(load, files)))
        y = np.array([labels[p.name] for p in files], dtype=np.uint8)
        np.savez(CACHE / f"codebrim_{split}.npz", x=x, y=y, names=np.array([p.name for p in files]))
        print(split, x.shape, "positives per class", y.sum(0).tolist(), "background", int((y.sum(1) == 0).sum()))
