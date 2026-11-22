"""Train one multi-label CODEBRIM model (official split) and evaluate on test.

Examples
  python experiments/train_codebrim.py --model convnext_tiny --epochs 30 --lr 2e-4 --seed 0
  python experiments/train_codebrim.py --model nano32 --epochs 60 --lr 3e-3 --seed 0 \
      --teacher convnext_tiny_r224_s0

Writes results_v2/vision/codebrim_<tag>.json, models/codebrim_<tag>.pt and the
train/val/test probabilities to <cache>/preds/codebrim_<tag>.npz.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectswarm import vision
from inspectswarm.classes import CODEBRIM_DEFECTS
from inspectswarm.manifest import save_result, seed_everything
from inspectswarm.paths import CACHE, RESULTS, WEIGHTS


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(vision.MODEL_ZOO))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--res", type=int, default=0)
    ap.add_argument("--teacher", default="", help="tag of a trained model whose train probabilities are the soft targets")
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--no-pretrain", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    cfg = vars(a).copy()
    res = a.res or vision.MODEL_ZOO[a.model][1]
    cfg["res"] = res
    tag = f"{a.model}_r{res}" + (f"_kd-{a.teacher}" if a.teacher else "") + f"_s{a.seed}"
    seed_everything(a.seed)
    device = torch.device(a.device)
    d = {s: np.load(CACHE / f"codebrim_{s}.npz") for s in ("train", "val", "test")}
    x = {s: d[s]["x"] for s in d}
    y = {s: d[s]["y"].astype(int) for s in d}

    teacher_prob = None
    if a.teacher:
        teacher_prob = np.load(CACHE / "preds" / f"codebrim_{a.teacher}.npz")["train"]

    model = vision.build_model(a.model, len(CODEBRIM_DEFECTS), pretrained=not a.no_pretrain)
    n_params = vision.count_params(model)
    print(f"{tag}: {n_params / 1e6:.3f} M params, res {res}, device {device}")
    hist = vision.train_multilabel(model, x["train"], y["train"], x["val"], y["val"], res=res, device=device,
                                   epochs=a.epochs, lr=a.lr, batch=a.batch, seed=a.seed,
                                   teacher_prob=teacher_prob, kd_alpha=a.kd_alpha)
    probs = {s: sigmoid(vision.predict_logits(model, x[s], res, device)) for s in x}
    result = {"tag": tag, "params": n_params, "history": hist,
              "val": vision.multilabel_metrics(y["val"], probs["val"], CODEBRIM_DEFECTS),
              "test": vision.multilabel_metrics(y["test"], probs["test"], CODEBRIM_DEFECTS)}
    print("TEST multi-target acc", round(result["test"]["multi_target_acc"], 4),
          "macro F1", round(result["test"]["macro_f1"], 4))
    (CACHE / "preds").mkdir(exist_ok=True)
    np.savez(CACHE / "preds" / f"codebrim_{tag}.npz", **probs)
    WEIGHTS.mkdir(exist_ok=True)
    torch.save(model.state_dict(), WEIGHTS / f"codebrim_{tag}.pt")
    save_result(RESULTS / "vision" / f"codebrim_{tag}.json", result, cfg, a.seed)
