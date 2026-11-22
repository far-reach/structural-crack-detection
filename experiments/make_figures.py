"""Build every figure of the paper from the JSON files in results_v2/.

Usage: python experiments/make_figures.py [--vision TAG --ut TAG]
Figures are written to results_v2/figures/ as PNG (300 dpi).
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inspectswarm.paths import CACHE, RESULTS

FIG = RESULTS / "figures"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans"})
C = {"vision": "#2b6cb0", "ut": "#c05621", "swarm": "#2f855a", "grey": "#4a5568", "light": "#edf2f7"}


def load(rel):
    return json.loads((RESULTS / rel).read_text())["result"]


# ------------------------------------------------------------- Fig. 1

def fig_framework():
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.set_xlim(0, 100), ax.set_ylim(0, 50), ax.axis("off")

    def box(x, y, w, h, title, lines, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2",
                                    fc="white", ec=color, lw=1.6))
        ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", weight="bold", color=color, fontsize=7.6)
        ax.text(x + w / 2, y + h - 7.0, "\n".join(lines), ha="center", va="top", fontsize=6.4, color="#1a202c",
                linespacing=1.35)

    def arrow(x0, y0, x1, y1, label="", dy=1.6):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1.2, color=C["grey"]))
        if label:
            ax.text((x0 + x1) / 2, (y0 + y1) / 2 + dy, label, ha="center", fontsize=6.8, color=C["grey"])

    box(1, 24, 27, 24, "Tier 1: screen", ["nano-drone scouts (simulated)", "camera image per zone",
                                          "small CNN on board", "output: P(visible damage)"], C["vision"])
    box(37, 24, 26, 24, "Fusion and tasking", ["prior of hidden defect", "from vision evidence",
                                               "select zones to confirm", "Bayesian update"], C["swarm"])
    box(72, 24, 27, 24, "Tier 2: confirm", ["contact node (simulated)", "single-point pulse-echo",
                                            "1D CNN on the A-scan", "output: P(hidden defect)"], C["ut"])
    arrow(28.6, 38, 36.4, 38, "evidence")
    arrow(63.6, 40, 71.4, 40, "tasks")
    arrow(71.4, 33, 63.6, 33, "evidence", dy=-3.4)
    box(1, 2, 27, 15, "Measured on real data", ["CODEBRIM images", "(held-out test regions)"], C["vision"])
    box(37, 2, 26, 15, "Assumed, then varied", ["platform parameters", "visible-to-hidden coupling"], C["swarm"])
    box(72, 2, 27, 15, "Measured on real data", ["BAM pulse-echo A-scans", "(held-out specimen)"], C["ut"])
    for x in (14.5, 50, 85.5):
        arrow(x, 17.6, x, 23.4)
    fig.savefig(FIG / "fig1_framework.png")
    plt.close(fig)


# ------------------------------------------------------------- vision

def vision_table():
    """Aggregate all CODEBRIM runs: mean and std over seeds per configuration."""
    rows = {}
    for f in sorted(glob.glob(str(RESULTS / "vision" / "codebrim_*.json"))):
        r = json.loads(Path(f).read_text())["result"]
        key = r["tag"].rsplit("_s", 1)[0]
        key = key.split("_kd-")[0] + ("+KD" if "_kd-" in key else "")
        rows.setdefault(key, {"params": r["params"], "runs": []})["runs"].append(r["test"])
    out = {}
    for k, v in rows.items():
        def ms(name):
            a = np.array([run[name] for run in v["runs"]])
            return float(a.mean()), float(a.std())

        out[k] = {"params": v["params"], "n": len(v["runs"]), "multi_target_acc": ms("multi_target_acc"),
                  "macro_f1": ms("macro_f1"), "damage_recall": ms("damage_recall"),
                  "false_alarm_rate": ms("false_alarm_rate"),
                  "per_class_f1": {c: float(np.mean([run["per_class"][c]["f1"] for run in v["runs"]]))
                                   for c in v["runs"][0]["per_class"]}}
    (RESULTS / "vision" / "summary.json").write_text(json.dumps(out, indent=1))
    return out


NAMES = {"convnext_tiny": "ConvNeXt-Tiny", "resnet50": "ResNet50", "mobilenetv3_small_100": "MobileNetV3-Small",
         "nano32": "NanoNet-32", "nano16": "NanoNet-16"}
LABEL_OFFSET = {"convnext_tiny": (-62, 6), "resnet50": (-44, -3), "mobilenetv3_small_100": (-80, -3),
                "nano32": (8, -3), "nano16": (8, -3)}


def fig_vision(summary, edge):
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for k, v in summary.items():
        kd = k.endswith("+KD")
        base = k.replace("+KD", "").rsplit("_r", 1)[0]
        m, s = v["multi_target_acc"]
        x = v["params"] / 1e6 * (1.08 if kd else 0.93)  # small horizontal shift so the pair does not overlap
        ax[0].errorbar(x, 100 * m, yerr=100 * s, fmt="o" if kd else "s", ms=5, capsize=2,
                       color=C["vision"] if kd else C["grey"], mfc=C["vision"] if kd else "white")
        if not kd:
            ax[0].annotate(NAMES[base], (v["params"] / 1e6, 100 * m), textcoords="offset points",
                           xytext=LABEL_OFFSET[base], fontsize=6.5)
    ax[0].axhline(72.19, color=C["ut"], ls="--", lw=1)
    ax[0].text(0.023, 72.6, "best of Mundt et al. (2019),\ntrained from scratch", color=C["ut"], fontsize=6, va="bottom")
    ax[0].set_xscale("log"), ax[0].set_xlim(0.02, 60), ax[0].set_ylim(63, 83)
    ax[0].set_xlabel("parameters (millions)"), ax[0].set_ylabel("multi-target accuracy, test (%)")
    ax[0].plot([], [], "o", color=C["vision"], label="distilled"), ax[0].plot([], [], "s", color=C["grey"], mfc="white", label="labels only")
    ax[0].legend(frameon=False, fontsize=7, loc="lower right")
    tags = [t for t in edge if "_kd-" not in t]
    tags.sort(key=lambda t: -edge[t]["fp32"]["size_kb"])
    xs = np.arange(len(tags))
    f = [100 * edge[t]["fp32"]["multi_target_acc"] for t in tags]
    q = [100 * edge[t]["int8"]["multi_target_acc"] for t in tags]
    ax[1].bar(xs - 0.2, f, 0.4, color=C["grey"], label="32-bit float")
    ax[1].bar(xs + 0.2, q, 0.4, color=C["vision"], label="8-bit integer")
    for i, t in enumerate(tags):
        ax[1].text(i + 0.2, q[i] + 1, f"{edge[t]['int8']['size_kb'] / 1024:.2f} MB", ha="center", fontsize=5.6, rotation=90, va="bottom")
    ax[1].set_xticks(xs), ax[1].set_xticklabels([NAMES[t.rsplit("_r", 1)[0]].replace("-Small", "-S").replace("-Tiny", "-T") for t in tags], fontsize=6, rotation=20)
    ax[1].set_ylim(0, 100), ax[1].set_ylabel("multi-target accuracy, test (%)")
    ax[1].legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    for a, t in zip(ax, "ab"):
        a.set_title(f"({t})", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_vision.png")
    plt.close(fig)


# ---------------------------------------------------------- ultrasonic

def fig_ultrasonic(ut_tag):
    from inspectswarm import ultrasonic as U

    wave, rot = ut_tag.split("_Rot")
    p = np.load(CACHE / "preds" / f"ascan_{ut_tag}.npz")
    fig = plt.figure(figsize=(7.2, 6.2))
    gs = fig.add_gridspec(3, 4, width_ratios=[1, 1, 1, 0.05], height_ratios=[1.2, 1, 1],
                          hspace=0.32, wspace=0.22, left=0.11, right=0.92, top=0.94, bottom=0.08)
    ax = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(3)])
    names = {"pk050": "Pk050 (plain)", "pk266": "Pk266 (tendon ducts)", "pk401": "Pk401 (foam cuboids)"}
    for j, s in enumerate(("pk050", "pk266", "pk401")):
        env = U.envelope(U.bandpass(U.load_volume(s, wave, rot), wave))
        row = 50 if s == "pk401" else 40
        b = env[:, row, :1700]
        ax[0, j].imshow((b / (b.max(1, keepdims=True) + 1e-9)).T ** 0.5, aspect="auto", cmap="magma", extent=[0, 2000, 850, 0])
        ax[0, j].set_title(f"{names[s]}\nB-scan at Y = {row * 10} mm", fontsize=8)
        lab, _ = U.object_labels(s)
        ax[1, j].imshow(np.where(lab.T == -1, 0.5, lab.T), aspect="auto", cmap="Greys", vmin=0, vmax=1, extent=[0, 2000, 800, 0])
        if not (lab == 1).any():
            ax[1, j].text(1000, 400, "no embedded objects", ha="center", va="center", fontsize=7, color=C["grey"])
        pm = np.where(p[f"valid_{s}"], p[f"p_{s}"], np.nan)
        im = ax[2, j].imshow(pm.T, aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=[0, 2000, 800, 0])
        for r in range(3):
            ax[r, j].set_xticks([0, 500, 1000, 1500, 2000])
            ax[r, j].tick_params(labelsize=7)
            if r < 2:
                ax[r, j].set_xticklabels([])
            if j > 0:
                ax[r, j].set_yticklabels([])
        ax[2, j].set_xlabel("X (mm)", fontsize=8)
    ax[0, 0].set_ylabel("B-scan\ntime (µs)", fontsize=8)
    ax[1, 0].set_ylabel("ground truth\nY (mm)", fontsize=8)
    ax[2, 0].set_ylabel("network output\nY (mm)", fontsize=8)
    ax[1, 2].text(2000, -40, "black: object   grey: excluded", ha="right", va="bottom", fontsize=6.5, color=C["grey"])
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[2, 3]))
    cb.set_label("P(embedded object)", fontsize=7), cb.ax.tick_params(labelsize=7)
    fig.savefig(FIG / "fig4_ultrasonic.png")
    plt.close(fig)


# --------------------------------------------------------------- swarm

def fig_swarm(sw):
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7))
    ns = (1, 2, 4, 8, 12, 16)
    for p_loss, col in zip((0.0, 0.1, 0.2), (C["swarm"], C["ut"], "#9b2c2c")):
        t = [sw["coverage_scaling"][f"loss{p_loss}_n{n}"]["time_min"]["mean"] for n in ns]
        ax[0].plot(ns, t, "o-", ms=3, color=col, label=f"loss per sortie {int(p_loss * 100)}%")
    ax[0].plot(ns, [sw["coverage_scaling"]["loss0.0_n1"]["time_min"]["mean"] / n for n in ns], ":", color=C["grey"], label="ideal 1/N")
    ax[0].set_xlabel("number of scouts"), ax[0].set_ylabel("tier 1 mission time (min)"), ax[0].set_yscale("log"), ax[0].set_xscale("log")
    ax[0].set_xticks([1, 2, 4, 8, 16]), ax[0].set_xticklabels([1, 2, 4, 8, 16]), ax[0].minorticks_off()
    ax[0].set_yticks([2, 5, 10, 20, 30]), ax[0].set_yticklabels([2, 5, 10, 20, 30]), ax[0].legend(frameon=False, fontsize=5.5, loc="upper right")
    pol = sw["confirm_policies"]
    names = {"vision_only": ("vision only", (4, 3)), "random_same_budget": ("random, same budget", (5, -2)),
             "flagged": ("flagged", (5, -9)), "flagged_plus_10pct_audit": ("flagged + 10% audit", (5, 4)),
             "exhaustive": ("exhaustive", (-38, -11)), "oracle_surface_truth": ("perfect scout", (-22, 12))}
    for k, (lab, off) in names.items():
        r = pol[k]
        ax[1].errorbar(r["contacts"]["mean"], 100 * r["hidden_recall"]["mean"], yerr=100 * r["hidden_recall"]["std"], fmt="o", ms=4, capsize=2)
        ax[1].annotate(lab, (r["contacts"]["mean"], 100 * r["hidden_recall"]["mean"]), textcoords="offset points", xytext=off, fontsize=6)
    ax[1].set_xlabel("ultrasonic contacts per mission"), ax[1].set_ylabel("hidden-defect recall (%)"), ax[1].set_ylim(-5, 110), ax[1].set_xlim(-120, 1650)
    noise = sw["sensitivity"]["ut_noise"]
    xs = sorted(float(k) for k in noise)
    for k, lab, col in (("flagged", "flagged", C["swarm"]), ("exhaustive", "exhaustive", C["ut"])):
        ax[2].plot(xs, [100 * noise[str(x)][k]["hidden_f1"]["mean"] for x in xs], "o-", ms=3, color=col, label=lab)
    ax[2].set_xlabel("noise blended into ultrasonic output"), ax[2].set_ylabel("hidden-defect F1 (%)"), ax[2].legend(frameon=False, fontsize=7)
    for a, t in zip(ax, "abc"):
        a.set_title(f"({t})", loc="left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_swarm.png")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ut", default="Shear_Rot00")
    ap.add_argument("--swarm", default="")
    a = ap.parse_args()
    fig_framework()
    summary = vision_table()
    print(json.dumps({k: (round(100 * v["multi_target_acc"][0], 2), round(100 * v["multi_target_acc"][1], 2), v["n"]) for k, v in summary.items()}, indent=0))
    if (RESULTS / "vision" / "edge_benchmark.json").exists():
        fig_vision(summary, load("vision/edge_benchmark.json"))
    if (CACHE / "preds" / f"ascan_{a.ut}.npz").exists():
        fig_ultrasonic(a.ut)
    sw = sorted(glob.glob(str(RESULTS / "swarm" / (a.swarm or "swarm_*.json"))))
    if sw:
        fig_swarm(json.loads(Path(sw[0]).read_text())["result"])
    print("figures in", FIG)
