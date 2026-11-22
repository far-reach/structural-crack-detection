"""Vision tier: models, GPU-side augmentation, training and metrics.

Device-agnostic: everything runs on CPU if CUDA is absent, only slower.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class NanoNet(nn.Module):
    """Depthwise-separable CNN sized for microcontroller-class processors.

    width=16 gives about 0.04 M parameters, width=32 about 0.14 M.
    """

    def __init__(self, num_outputs: int, width: int = 24):
        super().__init__()
        w = width

        def block(i, o, s):
            return nn.Sequential(
                nn.Conv2d(i, i, 3, s, 1, groups=i, bias=False), nn.BatchNorm2d(i), nn.ReLU6(inplace=True),
                nn.Conv2d(i, o, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU6(inplace=True))

        self.features = nn.Sequential(
            nn.Conv2d(3, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU6(inplace=True),
            block(w, 2 * w, 2), block(2 * w, 2 * w, 1),
            block(2 * w, 4 * w, 2), block(4 * w, 4 * w, 1),
            block(4 * w, 8 * w, 2), block(8 * w, 8 * w, 1))
        self.head = nn.Linear(8 * w, num_outputs)

    def forward(self, x):
        return self.head(F.adaptive_avg_pool2d(self.features(x), 1).flatten(1))


# name -> (constructor kind, default input resolution)
MODEL_ZOO = {
    "convnext_tiny": ("timm", 224),
    "resnet50": ("timm", 224),
    "mobilenetv3_small_100": ("timm", 224),
    "nano32": ("nano", 128),
    "nano16": ("nano", 96),
}


def build_model(name: str, num_outputs: int, pretrained: bool = True) -> nn.Module:
    kind, _ = MODEL_ZOO[name]
    if kind == "nano":
        return NanoNet(num_outputs, width=int(name[4:]))
    import timm

    return timm.create_model(name, pretrained=pretrained, num_classes=num_outputs)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def normalise(x_uint8: torch.Tensor, res: int) -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,res,res) float, ImageNet-normalised, no augmentation."""
    x = x_uint8.permute(0, 3, 1, 2).float() / 255.0
    if x.shape[-1] != res:
        x = F.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
    mean = torch.tensor(MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def augment(x_uint8: torch.Tensor, res: int, gen: torch.Generator) -> torch.Tensor:
    """Per-sample random crop (60 to 100% area), flips, 90-degree rotation and
    brightness/contrast jitter, all on the tensor's device."""
    n, dev = x_uint8.shape[0], x_uint8.device
    x = x_uint8.permute(0, 3, 1, 2).float() / 255.0

    def rnd(*shape):
        return torch.rand(*shape, generator=gen, device=dev)

    scale = torch.sqrt(0.6 + 0.4 * rnd(n))
    tx, ty = (1 - scale) * (2 * rnd(n) - 1), (1 - scale) * (2 * rnd(n) - 1)
    fx = torch.where(rnd(n) < 0.5, -1.0, 1.0)
    fy = torch.where(rnd(n) < 0.5, -1.0, 1.0)
    swap = rnd(n) < 0.5
    theta = torch.zeros(n, 2, 3, device=dev)
    a, b = scale * fx, scale * fy
    theta[:, 0, 0] = torch.where(swap, torch.zeros_like(a), a)
    theta[:, 0, 1] = torch.where(swap, a, torch.zeros_like(a))
    theta[:, 1, 0] = torch.where(swap, b, torch.zeros_like(b))
    theta[:, 1, 1] = torch.where(swap, torch.zeros_like(b), b)
    theta[:, 0, 2], theta[:, 1, 2] = tx, ty
    grid = F.affine_grid(theta, (n, 3, res, res), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    bright = 1 + 0.2 * (2 * rnd(n, 1, 1, 1) - 1)
    contrast = 1 + 0.2 * (2 * rnd(n, 1, 1, 1) - 1)
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = ((x - m) * contrast + m) * bright
    x = x.clamp(0, 1)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def predict_logits(model, x_uint8, res, device, batch=128) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(x_uint8), batch):
        xb = torch.as_tensor(x_uint8[i:i + batch]).to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(normalise(xb, res)).float().cpu())
    return torch.cat(out).numpy()


def multilabel_metrics(y: np.ndarray, prob: np.ndarray, names, thr: float = 0.5) -> dict:
    """Exact-match (multi-target) accuracy, per-class and macro F1."""
    pred = (prob >= thr).astype(int)
    res = {"multi_target_acc": float((pred == y).all(1).mean()), "per_class": {}}
    f1s = []
    for j, n in enumerate(names):
        tp = int(((pred[:, j] == 1) & (y[:, j] == 1)).sum())
        fp = int(((pred[:, j] == 1) & (y[:, j] == 0)).sum())
        fn = int(((pred[:, j] == 0) & (y[:, j] == 1)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        res["per_class"][n] = {"precision": p, "recall": r, "f1": f1, "support": int(y[:, j].sum()),
                               "acc": float((pred[:, j] == y[:, j]).mean())}
    res["macro_f1"] = float(np.mean(f1s))
    res["mean_class_acc"] = float(np.mean([v["acc"] for v in res["per_class"].values()]))
    bg_true, bg_pred = y.sum(1) == 0, pred.sum(1) == 0
    res["damage_recall"] = float((~bg_pred[~bg_true]).mean())
    res["false_alarm_rate"] = float((~bg_pred[bg_true]).mean()) if bg_true.any() else float("nan")
    return res


def train_multilabel(model, xtr, ytr, xva, yva, *, res, device, epochs, lr, batch=64, seed=0,
                     teacher_prob=None, kd_alpha=0.5, weight_decay=0.05, log=print):
    """AdamW + cosine schedule, BCE loss. If `teacher_prob` (N,C) is given, the
    loss mixes ground truth with the teacher's soft targets (knowledge
    distillation for multi-label outputs). Returns the best-on-validation
    state dict, selected by multi-target accuracy."""
    gen = torch.Generator(device=device).manual_seed(seed)
    cpu_gen = torch.Generator().manual_seed(seed)
    model.to(device)
    xtr_t = torch.as_tensor(xtr).to(device)
    ytr_t = torch.as_tensor(ytr).float().to(device)
    tp_t = torch.as_tensor(teacher_prob).float().to(device) if teacher_prob is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    steps = epochs * (len(xtr) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, tot = time.time(), 0.0
        perm = torch.randperm(len(xtr), generator=cpu_gen).to(device)
        for i in range(0, len(perm) - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = augment(xtr_t[idx], res, gen)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logit = model(xb)
                loss = F.binary_cross_entropy_with_logits(logit.float(), ytr_t[idx])
                if tp_t is not None:
                    kd = F.binary_cross_entropy_with_logits(logit.float(), tp_t[idx])
                    loss = (1 - kd_alpha) * loss + kd_alpha * kd
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
        prob = 1 / (1 + np.exp(-predict_logits(model, xva, res, device)))
        acc = float(((prob >= 0.5).astype(int) == yva).all(1).mean())
        history.append({"epoch": ep + 1, "loss": tot / max(1, len(perm) // batch), "val_multi_target_acc": acc})
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        log(f"  ep {ep + 1:02d} loss {history[-1]['loss']:.4f} val_mt {acc:.4f} best {best:.4f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    return history
