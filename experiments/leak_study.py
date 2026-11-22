"""How much of the v1 Stage 1 accuracy is explained by the evaluation protocol?

Trains the v1 recipe (ResNet50, layers 3-4 at 1e-5, head at 1e-3, Adam,
StepLR(7, 0.1), 20 epochs, batch 16, class-weighted cross-entropy) under two
protocols and evaluates each model on both validation sets:

  tile    : the exact v1 split (rebuilt): SDNET2018 tiles split at random, so
            every validation tile has its parent photograph in training;
            validation contains offline-augmented corrosion and spalling copies.
  grouped : SDNET2018 parent photographs split 80/20 (stratified by structure
            type); no augmented image in validation; corrosion and spalling
            evaluated on the original Raidathmane val + test patches.

Usage: python experiments/leak_study.py --seeds 0 1 2
"""
import argparse
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectswarm import vision
from inspectswarm.classes import V1_CLASSES
from inspectswarm.manifest import save_result, seed_everything
from inspectswarm.paths import RAIDATHMANE, RESULTS, ROOT, SDNET

REBUILD = ROOT / "rebuild" / "codebrim"
IDX = {c: i for i, c in enumerate(V1_CLASSES)}


def parent_of(path: Path) -> str:
    name = path.name
    if name.startswith("aug_"):
        name = name.split("_", 2)[2]
    return name.replace("-", "_").split("_")[0]


def raid_class(label_path: Path):
    a = np.array(Image.open(label_path).convert("RGB"))
    c = int(np.all(a == (255, 0, 0), axis=2).sum())
    s = int(np.all(a == (255, 255, 0), axis=2).sum())
    if c >= 100 and c >= s:
        return "corrosion"
    if s >= 100:
        return "spalling"
    return None


def raid_items(split: str):
    out = []
    for lab in sorted((RAIDATHMANE / split).glob("*_lab.png")):
        img = Path(str(lab).replace("_lab.png", ".png"))
        cls = raid_class(lab)
        if cls and img.exists():
            out.append((img, cls))
    return out


def tile_protocol():
    """The rebuilt v1 folders, exactly as used in the preprint."""
    tr = [(p, c) for c in V1_CLASSES for p in sorted((REBUILD / "train" / c).iterdir())]
    va = [(p, c) for c in V1_CLASSES for p in sorted((REBUILD / "val" / c).iterdir())]
    return tr, va


def grouped_protocol(seed: int):
    rng = random.Random(seed)
    train_par, val_par = set(), set()
    for structure in ("Decks", "Walls", "Pavements"):
        pars = sorted({parent_of(p) for p in (SDNET / structure).rglob("*.jpg")})
        rng.shuffle(pars)
        k = round(0.8 * len(pars))
        train_par |= {(structure, p) for p in pars[:k]}
        val_par |= {(structure, p) for p in pars[k:]}
    pools = {("train", "crack"): [], ("train", "intact"): [], ("val", "crack"): [], ("val", "intact"): []}
    for structure in ("Decks", "Walls", "Pavements"):
        for sub in sorted((SDNET / structure).iterdir()):
            cls = "intact" if sub.name.lower().startswith("n") else "crack"
            for p in sorted(sub.glob("*.jpg")):
                phase = "train" if (structure, parent_of(p)) in train_par else "val"
                pools[(phase, cls)].append(p)
    tr, va = [], []
    for (phase, cls), files in pools.items():
        rng.shuffle(files)
        n = 2400 if phase == "train" else 600
        (tr if phase == "train" else va).extend((p, cls) for p in files[:n])
    tr += raid_items("train")
    va += raid_items("val") + raid_items("test")
    return tr, va, {"train_parents": len(train_par), "val_parents": len(val_par)}


def load_images(items):
    def load(it):
        return np.asarray(Image.open(it[0]).convert("RGB").resize((224, 224), Image.BILINEAR))

    with ThreadPoolExecutor(16) as ex:
        x = np.stack(list(ex.map(load, items)))
    return x, np.array([IDX[c] for _, c in items])


def v1_model():
    m = models.resnet50(weights="IMAGENET1K_V1")
    for name, p in m.named_parameters():
        p.requires_grad = "layer3" in name or "layer4" in name
    m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 256), nn.ReLU(),
                         nn.Dropout(0.3), nn.Linear(256, 4))
    return m


def train(x, y, seed, device, epochs=20, batch=16):
    seed_everything(seed)
    m = v1_model().to(device)
    counts = np.bincount(y, minlength=4)
    w = torch.tensor(1.0 / counts, dtype=torch.float)
    w = (w / w.sum() * 4).to(device)
    opt = torch.optim.Adam([{"params": m.layer3.parameters(), "lr": 1e-5},
                            {"params": m.layer4.parameters(), "lr": 1e-5},
                            {"params": m.fc.parameters(), "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.StepLR(opt, 7, 0.1)
    crit = nn.CrossEntropyLoss(weight=w)
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    xt, yt = torch.as_tensor(x).to(device), torch.as_tensor(y).to(device)
    for ep in range(epochs):
        m.train()
        perm = torch.randperm(len(x), generator=cpu_gen).to(device)
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            if len(idx) < 2:
                continue
            loss = crit(m(vision.augment(xt[idx], 224, gen)), yt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        sched.step()
    return m  # last epoch: no selection on a validation set


def evaluate(m, x, y, items, device):
    pred = vision.predict_logits(m, x, 224, device).argmax(1)
    cm = np.zeros((4, 4), int)
    for t, p in zip(y, pred):
        cm[t, p] += 1
    rec = cm.diagonal() / np.maximum(cm.sum(1), 1)
    prec = cm.diagonal() / np.maximum(cm.sum(0), 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    sd = np.isin(y, [IDX["crack"], IDX["intact"]])
    aug = np.array([p.name.startswith("aug_") for p, _ in items])
    return {"n": int(len(y)), "acc": float((pred == y).mean()), "macro_f1": float(f1.mean()),
            "balanced_acc": float(rec.mean()), "f1": dict(zip(V1_CLASSES, f1.tolist())),
            "confusion": cm.tolist(),
            "acc_sdnet_source": float((pred == y)[sd].mean()),
            "acc_raidathmane_source": float((pred == y)[~sd].mean()),
            "acc_originals_only": float((pred == y)[~aug].mean()), "n_augmented": int(aug.sum())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    tile_tr, tile_va = tile_protocol()
    xt_tr, yt_tr = load_images(tile_tr)
    xt_va, yt_va = load_images(tile_va)
    print("tile protocol", len(tile_tr), len(tile_va))
    runs = []
    for seed in a.seeds:
        g_tr, g_va, info = grouped_protocol(seed)
        tr_par = {parent_of(p) for p, c in g_tr if c in ("crack", "intact")}
        assert not any(parent_of(p) in tr_par for p, c in g_va if c in ("crack", "intact")), "group leak"
        assert not any(p.name.startswith("aug_") for p, _ in g_va)
        xg_tr, yg_tr = load_images(g_tr)
        xg_va, yg_va = load_images(g_va)
        print(f"seed {seed}: grouped protocol {len(g_tr)} train, {len(g_va)} val, classes",
              np.bincount(yg_va).tolist(), info)
        m_tile = train(xt_tr, yt_tr, seed, device, a.epochs)
        m_group = train(xg_tr, yg_tr, seed, device, a.epochs)
        run = {"seed": seed, "grouped_info": info,
               "tile_model_on_tile_val": evaluate(m_tile, xt_va, yt_va, tile_va, device),
               "grouped_model_on_grouped_val": evaluate(m_group, xg_va, yg_va, g_va, device),
               # the tile-trained model saw the parents of the grouped val tiles: reported for reference only
               "tile_model_on_grouped_val_contaminated": evaluate(m_tile, xg_va, yg_va, g_va, device)}
        for k in ("tile_model_on_tile_val", "grouped_model_on_grouped_val"):
            print(f"  {k}: acc {run[k]['acc']:.4f} macroF1 {run[k]['macro_f1']:.4f} "
                  f"sdnet {run[k]['acc_sdnet_source']:.4f} raid {run[k]['acc_raidathmane_source']:.4f}")
        runs.append(run)
        del m_tile, m_group
    summary = {}
    for k in ("tile_model_on_tile_val", "grouped_model_on_grouped_val"):
        for metric in ("acc", "macro_f1", "acc_sdnet_source", "acc_raidathmane_source"):
            v = [r[k][metric] for r in runs]
            summary[f"{k}.{metric}"] = {"mean": float(np.mean(v)), "std": float(np.std(v)), "values": v}
    print({k: round(v["mean"], 4) for k, v in summary.items()})
    save_result(RESULTS / "vision" / "leak_study.json", {"runs": runs, "summary": summary}, vars(a), a.seeds)
