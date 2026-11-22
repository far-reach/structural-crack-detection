"""Check which output index the released Stage 1 model fires for each true class."""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from inspectswarm.classes import V1_CLASSES, V1_WRONG_DECODE_ORDER
from inspectswarm.paths import ROOT as _ROOT, REPO as _REPO, RESULTS as _RESULTS

import random, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms

ROOT = _ROOT
DS = ROOT / "datasets"
random.seed(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
m = models.resnet50(weights=None)
m.fc = nn.Sequential(nn.Dropout(0.4), nn.Linear(m.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4))
m.load_state_dict(torch.load(_REPO / "models/best_resnet50_multiclass.pth", map_location=dev))
m.eval().to(dev)
tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

crack, intact = [], []
for s in ["Decks", "Walls", "Pavements"]:
    for sub in (DS / "sdnet2018" / s).iterdir():
        (intact if sub.name.lower().startswith(("u","n")) else crack).extend(sub.glob("*.jpg"))
corr, spal = [], []
for lab in (DS / "raidathmane").rglob("*_lab.png"):
    a = np.array(Image.open(lab).convert("RGB"))
    c = int(np.all(a == (255, 0, 0), axis=2).sum()); s = int(np.all(a == (255, 255, 0), axis=2).sum())
    img = Path(str(lab).replace("_lab.png", ".png"))
    if not img.exists(): continue
    if c >= 100 and c >= s: corr.append(img)
    elif s >= 100: spal.append(img)
print("pool sizes", len(crack), len(intact), len(corr), len(spal))
N = 300
for name, pool in [("crack", crack), ("intact", intact), ("corrosion", corr), ("spalling", spal)]:
    pick = random.sample(pool, min(N, len(pool)))
    counts = np.zeros(4, int)
    with torch.no_grad():
        for i in range(0, len(pick), 64):
            x = torch.stack([tf(Image.open(p).convert("RGB")) for p in pick[i:i+64]]).to(dev)
            for k in m(x).argmax(1).tolist(): counts[k] += 1
    print(f"true {name:10s} n={len(pick)} output-index counts [0,1,2,3] = {counts.tolist()}")
