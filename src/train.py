"""
Structural Crack Detection - Training Script
Trains ResNet50 transfer learning model on concrete crack dataset.

Usage:
    python src/train.py --data_dir ./data --epochs 10 --batch_size 32

Expected to reach ~99% accuracy on Kaggle concrete crack dataset.
"""

import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from model import build_model
from dataset import get_dataloaders, CLASSES


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, 100.0 * correct / total


def val_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return running_loss / total, 100.0 * correct / total, all_preds, all_labels


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Structural Crack Detection - Training Results", fontsize=14, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", linewidth=2)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-o", label="Val Acc", linewidth=2)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([80, 101])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Training curves saved: {save_path}")


def plot_confusion_matrix(labels, preds, save_path):
    cm = confusion_matrix(labels, preds)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No Crack", "Crack"],
        yticklabels=["No Crack", "Crack"],
        ax=ax
    )
    ax.set_title("Confusion Matrix - Crack Detection", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Confusion matrix saved: {save_path}")


def main(args):
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, val_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size
    )

    # Model
    model = build_model(num_classes=2, freeze_backbone=True)
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    print(f"\nTraining for {args.epochs} epochs...\n")
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>8} | {'Val Acc':>8}")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_preds, val_labels = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"{epoch:>6} | {train_loss:>10.4f} | {train_acc:>8.2f}% | {val_loss:>8.4f} | {val_acc:>7.2f}%  ({elapsed:.1f}s)")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")
            print(f"         ✓ New best model saved ({best_val_acc:.2f}%)")

    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")

    # Plots
    plot_training_curves(history, f"{args.output_dir}/training_curves.png")
    plot_confusion_matrix(val_labels, val_preds, f"{args.output_dir}/confusion_matrix.png")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds, target_names=["No Crack", "Crack"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crack detection model")
    parser.add_argument("--data_dir",    type=str, default="./data",    help="Path to dataset")
    parser.add_argument("--output_dir",  type=str, default="./results", help="Output directory")
    parser.add_argument("--epochs",      type=int, default=10)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    args = parser.parse_args()
    main(args)
