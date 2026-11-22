"""Embedded-object detection from single pulse-echo A-scans, cross-specimen.

Train on two BAM specimens, test on the third, so the test specimen (and for
pk401 the object type) is never seen in training:

  test pk401 : train pk050 (plain) + pk266 (tendon ducts) -> polystyrene cuboids
  test pk266 : train pk050 (plain) + pk401 (cuboids)      -> tendon ducts
  test pk050 : train pk266 + pk401                         -> plain concrete, false alarms only

No model selection on test data: fixed epochs, last epoch is reported.

Usage: python experiments/train_ultrasonic.py --wave Shear --rot 00 --seeds 0 1 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectswarm import ultrasonic as U
from inspectswarm.manifest import save_result, seed_everything
from inspectswarm.paths import CACHE, RESULTS, WEIGHTS

SPECIMENS = ("pk050", "pk266", "pk401")
GATE_US = {"Long": 85.0, "Shear": 110.0}
MAX_US = {"Long": 700.0, "Shear": 900.0}


def prepare(wave, rot):
    out = {}
    for s in SPECIMENS:
        v = U.load_volume(s, wave, rot)
        f = U.bandpass(v, wave)
        lab, cover = U.object_labels(s)
        out[s] = {"x": U.traces_for_model(f, GATE_US[wave]), "env": U.envelope(f), "label": lab, "cover": cover,
                  "valid": U.valid_mask(v)}
        print(s, wave, rot, "valid", int(out[s]["valid"].sum()), "pos", int(((lab == 1) & out[s]["valid"]).sum()),
              "ignore", int(((lab == -1) & out[s]["valid"]).sum()))
    return out


def flat(d, names):
    """Training arrays, plus a mask of the calibration blocks: one in four
    100 mm blocks of each training specimen is withheld from training and
    used only to calibrate the output probabilities."""
    xs, ys, cs = [], [], []
    ix, iy = np.meshgrid(np.arange(201) // 10, np.arange(81) // 10, indexing="ij")
    calib_block = (ix + 2 * iy) % 4 == 0
    for s in names:
        m = d[s]["valid"] & (d[s]["label"] >= 0)
        xs.append(d[s]["x"][m])
        ys.append(d[s]["label"][m])
        cs.append(calib_block[m])
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(cs)


def platt_balanced(net, x_cal, y_cal, device):
    """Platt scaling fitted on the calibration blocks. Returns a function that
    maps raw outputs to probabilities at a 50% prior, i.e. the calibrated odds
    divided by the calibration prevalence odds, so that odds(p) is a
    likelihood ratio for the fusion rule."""
    from sklearn.linear_model import LogisticRegression

    def lg(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    z = lg(predict(net, x_cal, device))
    lr = LogisticRegression(C=1e6).fit(z[:, None], y_cal)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    prior = float(y_cal.mean())

    def apply(p_raw):
        return 1 / (1 + np.exp(-(a * lg(p_raw) + b - lg(prior))))

    return apply, {"a": a, "b": b, "calibration_prevalence": prior, "n": int(len(y_cal))}


def train(x, y, seed, device, epochs=30, batch=256, lr=2e-3):
    seed_everything(seed)
    net = U.build_ascan_net().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    steps = epochs * (len(x) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps, pct_start=0.1)
    xt, yt = torch.as_tensor(x).to(device), torch.as_tensor(y).float().to(device)
    pos_w = torch.tensor((1 - y.mean()) / y.mean(), device=device)
    g = torch.Generator(device=device).manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(x), generator=g, device=device)
        for i in range(0, len(perm) - batch + 1, batch):
            idx = perm[i:i + batch]
            xb = xt[idx]
            # augmentation: time stretch (other depth or wave speed), gain, additive noise
            xb = time_stretch(xb, 0.85 + 0.3 * float(torch.rand(1, generator=g, device=device)))
            xb = xb * (0.7 + 0.6 * torch.rand(len(idx), 1, 1, generator=g, device=device))
            xb = xb + 0.02 * torch.randn(xb.shape, generator=g, device=device)
            loss = F.binary_cross_entropy_with_logits(net(xb).squeeze(1), yt[idx], pos_weight=pos_w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
    return net


def time_stretch(xb, factor):
    """Resample a batch (N, C, L) along time by `factor`, then crop or zero-pad
    back to L. Mimics an echo from a different depth or a different wave speed."""
    n = xb.shape[-1]
    y = F.interpolate(xb, size=max(8, int(round(n * factor))), mode="linear", align_corners=False)
    return y[..., :n] if y.shape[-1] >= n else F.pad(y, (0, n - y.shape[-1]))


@torch.no_grad()
def predict(net, x, device):
    net.eval()
    xt = torch.as_tensor(x.reshape(-1, *x.shape[-2:])).to(device)
    out = torch.cat([torch.sigmoid(net(xt[i:i + 4096]).squeeze(1)) for i in range(0, len(xt), 4096)])
    return out.cpu().numpy().reshape(x.shape[:-2])


def score(y, p):
    pred = p >= 0.5
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    r = {"n": int(len(y)), "pos": int((y == 1).sum()), "specificity": tn / max(tn + fp, 1)}
    if (y == 1).any():
        rec, prec = tp / (tp + fn), tp / max(tp + fp, 1)
        r.update(recall=rec, precision=prec, f1=2 * prec * rec / max(prec + rec, 1e-9),
                 auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)))
    return r


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wave", default="Shear", choices=["Shear", "Long"])
    ap.add_argument("--rot", default="00", choices=["00", "90"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)
    d = prepare(a.wave, a.rot)
    tag = f"{a.wave}_Rot{a.rot}"

    # physics baseline, calibrated on pk050 only
    m0 = d["pk050"]["valid"]
    th0, clear0 = U.thickness_map("pk050", 60)
    cal0 = m0 & clear0 & (th0 != U.THICKNESS["pk050"][1])  # step 2 excluded, see ultrasonic_thickness.py
    c, t0 = U.fit_velocity(U.pick_backwall(d["pk050"]["env"], GATE_US[a.wave], MAX_US[a.wave])[cal0], th0[cal0])
    result = {"calibration_pk050": {"c_ms": c, "t0_us": t0}, "folds": {}}
    print(f"calibrated on pk050: c = {c:.0f} m/s, t0 = {t0:.1f} us")

    preds = {}
    for test in SPECIMENS:
        train_names = [s for s in SPECIMENS if s != test]
        xall, yall, cal = flat(d, train_names)
        xtr, ytr = xall[~cal], yall[~cal]
        m = d[test]["valid"] & (d[test]["label"] >= 0)
        yte = d[test]["label"][m]
        th, _ = U.thickness_map(test)
        base = U.classical_score(d[test]["env"], th, c, t0, GATE_US[a.wave])[m]
        fold = {"train": train_names, "train_n": int(len(ytr)), "train_pos_rate": float(ytr.mean()), "runs": [],
                "classical": ({"auroc": float(roc_auc_score(yte, base)), "ap": float(average_precision_score(yte, base))}
                              if (yte == 1).any() else {})}
        pm = []
        for seed in a.seeds:
            net = train(xtr, ytr, seed, device, a.epochs)
            calibrate, cal_info = platt_balanced(net, xall[cal], yall[cal], device)
            p_raw = predict(net, d[test]["x"], device)
            p_full = calibrate(p_raw)
            pm.append(p_full)
            r = score(yte, p_full[m])
            r["seed"] = seed
            r["calibration"] = cal_info
            r["uncalibrated"] = score(yte, p_raw[m])
            if (yte == 1).any():  # recall by object depth
                r["recall_by_cover_mm"] = {str(cv): float((p_full[(d[test]["cover"] == cv) & m] >= 0.5).mean())
                                           for cv in np.unique(d[test]["cover"][m & (d[test]["label"] == 1)])}
            fold["runs"].append(r)
            if seed == a.seeds[0]:
                WEIGHTS.mkdir(exist_ok=True)
                torch.save(net.state_dict(), WEIGHTS / f"ascan_{tag}_test-{test}_s{seed}.pt")
        for k in ("auroc", "ap", "f1", "recall", "precision", "specificity"):
            v = [r[k] for r in fold["runs"] if k in r]
            if v:
                fold[k] = {"mean": float(np.mean(v)), "std": float(np.std(v))}
        preds[test] = np.mean(pm, 0)
        result["folds"][test] = fold
        print(test, {k: round(fold[k]["mean"], 4) for k in ("auroc", "ap", "f1", "recall", "specificity") if k in fold},
              "classical", {k: round(v, 4) for k, v in fold["classical"].items()})
    (CACHE / "preds").mkdir(parents=True, exist_ok=True)
    np.savez(CACHE / "preds" / f"ascan_{tag}.npz", **{f"p_{s}": preds[s] for s in preds},
             **{f"label_{s}": d[s]["label"] for s in SPECIMENS}, **{f"valid_{s}": d[s]["valid"] for s in SPECIMENS})
    save_result(RESULTS / "ultrasonic" / f"detect_{tag}.json", result, vars(a), a.seeds)
