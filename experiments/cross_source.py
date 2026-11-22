"""Does a model trained on one image source hold on another?

A. CODEBRIM-trained multi-label models are tested on the original
   SDNET2018 tiles (600 per class, sampled at random) and the original
   Raidathmane val and test patches. A patch counts
   as flagged if any defect probability reaches 0.5; class agreement uses the
   mapping in inspectswarm.classes.V1_TO_CODEBRIM.
B. The released v1.0 classifier (trained on SDNET2018 + Raidathmane) is tested
   on CODEBRIM test crops that carry exactly one of its classes, or none.

Usage: python experiments/cross_source.py --tags convnext_tiny_r224_s0 ...
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.leak_study import load_images, raid_items  # noqa: E402
from inspectswarm import vision
from inspectswarm.classes import CODEBRIM_DEFECTS, V1_CLASSES, V1_TO_CODEBRIM
from inspectswarm.manifest import save_result
from inspectswarm.paths import CACHE, RESULTS, SDNET, WEIGHTS


DAMAGE = [c for c in V1_CLASSES if V1_TO_CODEBRIM[c]]  # the three damage classes


def external_set(seed=0, n_per_class=600):
    rng = random.Random(seed)
    items = []
    for cls, prefix in (("crack", "c"), ("intact", "n")):
        files = sorted(p for s in ("Decks", "Walls", "Pavements") for sub in (SDNET / s).iterdir()
                       if sub.name.lower().startswith(prefix) for p in sub.glob("*.jpg"))
        rng.shuffle(files)
        items += [(p, cls) for p in files[:n_per_class]]
    return items + raid_items("val") + raid_items("test")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    items = external_set()
    x, y = load_images(items)  # 224 px; y indexes V1_CLASSES
    cls_of = np.array(V1_CLASSES)[y]
    result = {"A_codebrim_models_on_sdnet_raidathmane": {}, "n_external": {c: int((cls_of == c).sum()) for c in V1_CLASSES}}
    for tag in a.tags:
        name, res = tag.split("_r")[0], int(tag.split("_r")[1].split("_")[0])
        m = vision.build_model(name, len(CODEBRIM_DEFECTS), pretrained=False)
        m.load_state_dict(torch.load(WEIGHTS / f"codebrim_{tag}.pt", map_location="cpu"))
        prob = 1 / (1 + np.exp(-vision.predict_logits(m.to(device), x, res, device)))
        flagged = (prob >= 0.5).any(1)
        row = {"false_alarm_rate_intact": float(flagged[cls_of == "intact"].mean())}
        for c in DAMAGE:
            cols = [CODEBRIM_DEFECTS.index(t) for t in V1_TO_CODEBRIM[c]]
            sel = cls_of == c
            row[f"flag_recall_{c}"] = float(flagged[sel].mean())
            row[f"class_recall_{c}"] = float((prob[sel][:, cols] >= 0.5).any(1).mean())
        dmg = cls_of != "intact"
        row["damage_recall"] = float(flagged[dmg].mean())
        row["balanced_acc_damage_vs_intact"] = float((flagged[dmg].mean() + (~flagged[~dmg]).mean()) / 2)
        result["A_codebrim_models_on_sdnet_raidathmane"][tag] = row
        print(tag, {k: round(v, 3) for k, v in row.items()})

    # B: v1.0 classifier on CODEBRIM test
    t = np.load(CACHE / "codebrim_test.npz")
    xt, yt = t["x"], t["y"].astype(int)
    v1 = models.resnet50(weights=None)
    v1.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(v1.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4))
    v1.load_state_dict(torch.load(WEIGHTS / "best_resnet50_multiclass.pth", map_location="cpu"))
    pred = np.array(V1_CLASSES)[vision.predict_logits(v1.to(device), xt, 224, device).argmax(1)]
    rowb = {}
    names = list(CODEBRIM_DEFECTS)
    bg = yt.sum(1) == 0
    rowb["intact_recall_on_background"] = float((pred[bg] == "intact").mean())
    rowb["n_background"] = int(bg.sum())
    for c in DAMAGE:
        cols = [names.index(tn) for tn in V1_TO_CODEBRIM[c]]
        others = [j for j in range(len(names)) if j not in cols]
        sel = (yt[:, cols].sum(1) > 0) & (yt[:, others].sum(1) == 0)
        rowb[f"recall_{c}"] = float((pred[sel] == c).mean())
        rowb[f"flagged_as_damage_{c}"] = float((pred[sel] != "intact").mean())
        rowb[f"n_{c}"] = int(sel.sum())
    dmg = ~bg
    rowb["damage_recall_all_defect_crops"] = float((pred[dmg] != "intact").mean())
    rowb["balanced_acc_damage_vs_intact"] = float(((pred[dmg] != "intact").mean() + (pred[bg] == "intact").mean()) / 2)
    rowb["prediction_share"] = {c: float((pred == c).mean()) for c in V1_CLASSES}
    result["B_v1_model_on_codebrim_test"] = rowb
    print("v1 on CODEBRIM", rowb)
    save_result(RESULTS / "vision" / "cross_source.json", result, vars(a))
