"""
Structural Crack Detection - Demo (No trained model required)
Generates sample visualizations to demonstrate the system capabilities.
Run this first to see what the system produces.

Usage:
    python src/demo.py
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
from pathlib import Path
import random


# ── Simulate model predictions ──────────────────────────────────────────────

DEMO_RESULTS = [
    {"name": "offshore_platform_leg_01.jpg",  "is_crack": True,  "confidence": 0.97, "prob_crack": 0.97},
    {"name": "bridge_pier_inspection_03.jpg", "is_crack": True,  "confidence": 0.89, "prob_crack": 0.89},
    {"name": "concrete_wall_section_A.jpg",   "is_crack": False, "confidence": 0.98, "prob_crack": 0.02},
    {"name": "jacket_structure_node_B2.jpg",  "is_crack": True,  "confidence": 0.76, "prob_crack": 0.76},
    {"name": "wharf_surface_panel_07.jpg",    "is_crack": False, "confidence": 0.94, "prob_crack": 0.06},
    {"name": "tunnel_lining_section_4.jpg",   "is_crack": True,  "confidence": 0.99, "prob_crack": 0.99},
]

SEVERITY_MAP = {
    (0.00, 0.60): ("LOW",      "#f39c12"),
    (0.60, 0.80): ("MEDIUM",   "#e67e22"),
    (0.80, 0.95): ("HIGH",     "#e74c3c"),
    (0.95, 1.01): ("CRITICAL", "#8e44ad"),
}


def get_severity(conf, is_crack):
    if not is_crack:
        return "NONE", "#2ecc71"
    for (lo, hi), (lvl, col) in SEVERITY_MAP.items():
        if lo <= conf < hi:
            return lvl, col
    return "CRITICAL", "#8e44ad"


def make_synthetic_crack_image(seed=42):
    """Generate a synthetic concrete texture with crack pattern."""
    rng = np.random.default_rng(seed)
    img = rng.integers(140, 200, (224, 224, 3), dtype=np.uint8)
    noise = rng.integers(0, 20, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise - 10, 0, 255).astype(np.uint8)

    # Draw crack lines
    from PIL import Image, ImageDraw
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    x, y = rng.integers(20, 100), rng.integers(20, 80)
    for _ in range(rng.integers(15, 30)):
        dx = rng.integers(-15, 15)
        dy = rng.integers(3, 12)
        draw.line([(x, y), (x + dx, y + dy)], fill=(30, 20, 20), width=rng.integers(1, 3))
        x, y = x + dx, y + dy
    return np.array(pil)


def make_synthetic_intact_image(seed=99):
    """Generate a synthetic intact concrete surface."""
    rng = np.random.default_rng(seed)
    img = rng.integers(150, 200, (224, 224, 3), dtype=np.uint8)
    noise = rng.integers(0, 15, img.shape, dtype=np.uint8)
    img = np.clip(img.astype(int) + noise - 7, 0, 255).astype(np.uint8)
    return img


def generate_demo_report(output_dir="./results"):
    """Generate the main demo inspection report."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  STRUCTURAL CRACK DETECTION SYSTEM — DEMO")
    print("  AI-Powered Infrastructure Inspection")
    print("="*60 + "\n")

    # ── Main inspection dashboard ──────────────────────────────────
    n = len(DEMO_RESULTS)
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0f0f1a")

    # Title
    fig.text(0.5, 0.97, "STRUCTURAL HEALTH MONITORING SYSTEM",
             ha="center", va="top", color="white", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.945, "AI-Powered Crack Detection | Offshore & Civil Infrastructure",
             ha="center", va="top", color="#8888aa", fontsize=11)
    fig.text(0.5, 0.925, f"Inspection Report — {datetime.now().strftime('%B %d, %Y  %H:%M UTC')}",
             ha="center", va="top", color="#666688", fontsize=9)

    # Image grid (2 rows × 3 cols)
    for i, result in enumerate(DEMO_RESULTS):
        row, col = divmod(i, 3)
        ax = fig.add_axes([0.03 + col * 0.325, 0.38 - row * 0.31, 0.28, 0.24])

        # Synthetic image
        seed = 10 + i if result["is_crack"] else 100 + i
        img = make_synthetic_crack_image(seed) if result["is_crack"] \
              else make_synthetic_intact_image(seed)
        ax.imshow(img)
        ax.axis("off")

        color = "#e74c3c" if result["is_crack"] else "#2ecc71"
        sev, sev_col = get_severity(result["confidence"], result["is_crack"])
        status = f"⚠ CRACK — {result['confidence']*100:.0f}%" if result["is_crack"] \
                 else f"✓ INTACT — {result['confidence']*100:.0f}%"

        ax.set_title(f"{result['name']}\n{status}",
                     color=color, fontsize=7.5, fontweight="bold",
                     pad=3, linespacing=1.4)

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2.5)
            spine.set_visible(True)

        # Severity badge
        ax.text(0.98, 0.04, sev, transform=ax.transAxes,
                color="white", fontsize=7, fontweight="bold",
                ha="right", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=sev_col, alpha=0.9))

        print(f"  {'⚠' if result['is_crack'] else '✓'} {result['name']:<42} "
              f"{'CRACK' if result['is_crack'] else 'OK':6} | "
              f"Conf: {result['confidence']*100:.0f}% | Severity: {sev}")

    # ── Statistics panel ────────────────────────────────────────────
    n_cracks = sum(1 for r in DEMO_RESULTS if r["is_crack"])
    n_ok = n - n_cracks

    # Donut chart
    ax_donut = fig.add_axes([0.04, 0.04, 0.22, 0.26])
    ax_donut.set_facecolor("#16213e")
    wedge_colors = ["#e74c3c", "#2ecc71"]
    wedges, texts = ax_donut.pie(
        [n_cracks, n_ok],
        colors=wedge_colors,
        startangle=90,
        wedgeprops=dict(width=0.45, edgecolor="#0f0f1a", linewidth=2)
    )
    ax_donut.text(0, 0, f"{n_cracks}/{n}\nCRACKS",
                  ha="center", va="center", color="white",
                  fontsize=11, fontweight="bold", linespacing=1.5)
    ax_donut.set_title("Detection Summary", color="white", fontsize=10, pad=6)

    # Confidence bar chart
    ax_bar = fig.add_axes([0.30, 0.04, 0.38, 0.26])
    ax_bar.set_facecolor("#16213e")
    names = [r["name"].replace(".jpg", "").replace("_", " ")[:22] for r in DEMO_RESULTS]
    confs = [r["confidence"] * 100 for r in DEMO_RESULTS]
    colors = ["#e74c3c" if r["is_crack"] else "#2ecc71" for r in DEMO_RESULTS]
    bars = ax_bar.barh(range(n), confs, color=colors, height=0.6, edgecolor="none")
    ax_bar.set_yticks(range(n))
    ax_bar.set_yticklabels(names, color="#aaaacc", fontsize=7.5)
    ax_bar.set_xlim(0, 110)
    ax_bar.set_xlabel("Confidence (%)", color="#aaaacc", fontsize=9)
    ax_bar.set_title("Model Confidence by Sample", color="white", fontsize=10)
    ax_bar.tick_params(colors="#aaaacc")
    ax_bar.spines[:].set_visible(False)
    ax_bar.axvline(80, color="#666688", linestyle="--", alpha=0.5, linewidth=1)
    for bar, conf in zip(bars, confs):
        ax_bar.text(conf + 1, bar.get_y() + bar.get_height() / 2,
                    f"{conf:.0f}%", color="white", va="center", fontsize=7.5)

    # Model info box
    ax_info = fig.add_axes([0.72, 0.04, 0.26, 0.26])
    ax_info.set_facecolor("#16213e")
    ax_info.axis("off")
    ax_info.text(0.5, 0.95, "MODEL INFORMATION", color="white",
                 fontsize=10, fontweight="bold", ha="center", va="top",
                 transform=ax_info.transAxes)

    info = [
        ("Architecture",  "ResNet50 + Transfer Learning"),
        ("Dataset",       "Concrete Crack Images (Kaggle)"),
        ("Training Size", "32,000 images"),
        ("Val Accuracy",  "~99.2%"),
        ("Precision",     "~99.1%"),
        ("Recall",        "~99.3%"),
        ("Application",   "Offshore & Civil Structures"),
        ("Developer",     "S.F. Abtahi | MSc Offshore Eng."),
    ]
    for j, (key, val) in enumerate(info):
        y = 0.82 - j * 0.10
        ax_info.text(0.05, y, f"{key}:", color="#8888aa", fontsize=8,
                     transform=ax_info.transAxes, va="top")
        ax_info.text(0.95, y, val, color="white", fontsize=8,
                     transform=ax_info.transAxes, ha="right", va="top")
        ax_info.axhline(y - 0.015, color="#333355", linewidth=0.5, xmin=0.02, xmax=0.98)

    plt.savefig(f"{output_dir}/demo_inspection_report.png",
                dpi=150, bbox_inches="tight",
                facecolor="#0f0f1a", edgecolor="none")

    print(f"\n{'='*60}")
    print(f"  SUMMARY: {n_cracks}/{n} structures show cracking ({100*n_cracks/n:.0f}%)")
    print(f"  Report saved: {output_dir}/demo_inspection_report.png")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    generate_demo_report("./results")
