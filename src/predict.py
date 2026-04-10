"""
Structural Crack Detection - Inference & Report Generation
Analyzes images for structural cracks and generates a visual inspection report.

Usage:
    # Single image
    python src/predict.py --model results/best_model.pth --image path/to/image.jpg

    # Folder of images
    python src/predict.py --model results/best_model.pth --folder path/to/images/

    # Generate full HTML report
    python src/predict.py --model results/best_model.pth --folder path/to/images/ --report
"""

import argparse
import os
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont

from model import build_model
from dataset import get_transforms, CLASSES, CLASS_COLORS


SEVERITY_LEVELS = {
    (0.0, 0.6):  ("LOW",      "#f39c12", "Monitor — no immediate action required"),
    (0.6, 0.80): ("MEDIUM",   "#e67e22", "Inspect within 30 days"),
    (0.80, 0.95):("HIGH",     "#e74c3c", "Priority inspection required"),
    (0.95, 1.01):("CRITICAL", "#8e44ad", "IMMEDIATE ACTION REQUIRED"),
}


def get_severity(confidence, is_crack):
    if not is_crack:
        return "NONE", "#2ecc71", "Structure appears intact"
    for (low, high), (level, color, message) in SEVERITY_LEVELS.items():
        if low <= confidence < high:
            return level, color, message
    return "CRITICAL", "#8e44ad", "IMMEDIATE ACTION REQUIRED"


def load_model(model_path, device):
    model = build_model(num_classes=2, freeze_backbone=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    model.to(device)
    return model


def predict_image(model, image_path, device):
    """Run inference on a single image."""
    transform = get_transforms("val")
    
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)
        confidence, predicted = probs.max(1)
    
    pred_class = predicted.item()
    conf = confidence.item()
    
    return {
        "image_path": str(image_path),
        "image": image,
        "predicted_class": pred_class,
        "class_name": CLASSES[pred_class],
        "confidence": conf,
        "prob_no_crack": probs[0][0].item(),
        "prob_crack": probs[0][1].item(),
        "is_crack": pred_class == 1,
    }


def visualize_single(result, save_path=None):
    """Create a single image inspection visualization."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")

    # Original image
    axes[0].imshow(result["image"])
    axes[0].set_title("Input Image", color="white", fontsize=13, pad=10)
    axes[0].axis("off")

    # Add border color based on result
    border_color = "#e74c3c" if result["is_crack"] else "#2ecc71"
    for spine in axes[0].spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
        spine.set_visible(True)

    # Results panel
    axes[1].set_facecolor("#16213e")
    axes[1].set_xlim(0, 10)
    axes[1].set_ylim(0, 10)
    axes[1].axis("off")

    severity, sev_color, sev_message = get_severity(result["confidence"], result["is_crack"])

    # Title
    axes[1].text(5, 9.2, "INSPECTION RESULT", color="white",
                 fontsize=14, fontweight="bold", ha="center", va="center")

    # Status box
    status_text = "⚠ CRACK DETECTED" if result["is_crack"] else "✓ NO CRACK DETECTED"
    axes[1].add_patch(patches.FancyBboxPatch(
        (0.5, 7.2), 9, 1.5, boxstyle="round,pad=0.1",
        facecolor=border_color, alpha=0.9, zorder=2
    ))
    axes[1].text(5, 7.95, status_text, color="white",
                 fontsize=16, fontweight="bold", ha="center", va="center", zorder=3)

    # Confidence bar
    axes[1].text(0.8, 6.6, "Confidence:", color="#aaaaaa", fontsize=11, va="center")
    axes[1].text(9.2, 6.6, f"{result['confidence']*100:.1f}%", 
                 color="white", fontsize=12, fontweight="bold", ha="right", va="center")
    
    bar_width = result["confidence"] * 8
    axes[1].add_patch(patches.Rectangle((0.8, 6.1), 8, 0.3, facecolor="#333355"))
    axes[1].add_patch(patches.Rectangle((0.8, 6.1), bar_width, 0.3, facecolor=border_color))

    # Probability breakdown
    axes[1].text(0.8, 5.4, "Probability Breakdown:", color="#aaaaaa", fontsize=11)
    
    axes[1].text(0.8, 4.8, "No Crack:", color="#2ecc71", fontsize=11)
    axes[1].text(9.2, 4.8, f"{result['prob_no_crack']*100:.1f}%",
                 color="#2ecc71", fontsize=11, fontweight="bold", ha="right")

    axes[1].text(0.8, 4.2, "Crack:", color="#e74c3c", fontsize=11)
    axes[1].text(9.2, 4.2, f"{result['prob_crack']*100:.1f}%",
                 color="#e74c3c", fontsize=11, fontweight="bold", ha="right")

    # Severity
    axes[1].add_patch(patches.FancyBboxPatch(
        (0.5, 2.8), 9, 1.0, boxstyle="round,pad=0.1",
        facecolor=sev_color, alpha=0.3, edgecolor=sev_color
    ))
    axes[1].text(5, 3.5, f"Severity: {severity}", color=sev_color,
                 fontsize=12, fontweight="bold", ha="center", va="center")
    axes[1].text(5, 2.9, sev_message, color="white",
                 fontsize=9, ha="center", va="center", alpha=0.9)

    # Timestamp
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    axes[1].text(5, 0.4, f"Analysis: {ts}", color="#666688",
                 fontsize=8, ha="center", va="center")
    axes[1].text(5, 0.1, "AI-Powered Structural Inspection System",
                 color="#444466", fontsize=7, ha="center", va="center")

    plt.suptitle("Structural Health Monitoring — Crack Detection Report",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor="#1a1a2e", edgecolor="none")
        print(f"Report saved: {save_path}")
    else:
        plt.show()
    plt.close()


def analyze_folder(model, folder_path, device, output_dir, report=False):
    """Analyze all images in a folder."""
    folder = Path(folder_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        p for p in folder.glob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
    ])

    if not image_files:
        print(f"No images found in {folder_path}")
        return

    print(f"\nAnalyzing {len(image_files)} images...\n")
    results = []

    for i, img_path in enumerate(image_files, 1):
        result = predict_image(model, img_path, device)
        results.append(result)

        status = "CRACK" if result["is_crack"] else "OK"
        print(f"[{i:3d}/{len(image_files)}] {img_path.name:<30} "
              f"{status:<6} | Conf: {result['confidence']*100:.1f}%")

        # Save individual result image
        save_path = output / f"{img_path.stem}_result.png"
        visualize_single(result, save_path=str(save_path))

    # Summary
    n_cracks = sum(1 for r in results if r["is_crack"])
    n_ok = len(results) - n_cracks
    crack_rate = 100 * n_cracks / len(results)

    print(f"\n{'='*50}")
    print(f"INSPECTION SUMMARY")
    print(f"{'='*50}")
    print(f"Total images analyzed: {len(results)}")
    print(f"Cracks detected:       {n_cracks} ({crack_rate:.1f}%)")
    print(f"No cracks:             {n_ok}")
    print(f"{'='*50}\n")

    if report:
        generate_summary_plot(results, output / "inspection_summary.png")

    return results


def generate_summary_plot(results, save_path):
    """Generate a summary dashboard of all analyzed images."""
    n = len(results)
    cols = min(5, n)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3 + 1))
    fig.patch.set_facecolor("#1a1a2e")
    fig.suptitle("Structural Inspection Summary", color="white",
                 fontsize=14, fontweight="bold", y=1.01)

    axes_flat = axes.flatten() if n > 1 else [axes]

    for i, result in enumerate(results):
        ax = axes_flat[i]
        ax.imshow(result["image"])
        
        color = "#e74c3c" if result["is_crack"] else "#2ecc71"
        label = f"CRACK {result['confidence']*100:.0f}%" if result["is_crack"] \
                else f"OK {result['confidence']*100:.0f}%"
        
        ax.set_title(label, color=color, fontsize=9, fontweight="bold", pad=4)
        ax.axis("off")
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(3)
            spine.set_visible(True)

    # Hide unused axes
    for j in range(len(results), len(axes_flat)):
        axes_flat[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight",
                facecolor="#1a1a2e", edgecolor="none")
    print(f"Summary plot saved: {save_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Structural crack detection inference")
    parser.add_argument("--model",  type=str, required=True, help="Path to trained model (.pth)")
    parser.add_argument("--image",  type=str, default=None,  help="Single image path")
    parser.add_argument("--folder", type=str, default=None,  help="Folder of images")
    parser.add_argument("--output", type=str, default="./results/inference", help="Output directory")
    parser.add_argument("--report", action="store_true", help="Generate summary report")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = load_model(args.model, device)
    print("Model loaded successfully.\n")

    if args.image:
        result = predict_image(model, args.image, device)
        print(f"Result:     {result['class_name']}")
        print(f"Confidence: {result['confidence']*100:.2f}%")
        severity, _, msg = get_severity(result["confidence"], result["is_crack"])
        print(f"Severity:   {severity} — {msg}")
        
        Path(args.output).mkdir(parents=True, exist_ok=True)
        img_name = Path(args.image).stem
        visualize_single(result, save_path=f"{args.output}/{img_name}_report.png")

    elif args.folder:
        analyze_folder(model, args.folder, device, args.output, report=args.report)

    else:
        print("Please provide --image or --folder argument.")
