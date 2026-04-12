"""
Stage 1: Multi-Class Structural Damage Detection
ResNet50 Transfer Learning → 4 Classes: crack, corrosion, spalling, intact
Author: Seyed Farhad Abtahi
"""

import os, time, copy, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_DIR    = "data/codebrim"
MODEL_DIR   = "models"
RESULTS_DIR = "results"
CLASSES     = ["crack", "intact", "corrosion", "spalling"]
NUM_CLASSES = 4
BATCH_SIZE  = 16
NUM_EPOCHS  = 20
LR          = 3e-4
IMG_SIZE    = 224
DEVICE      = torch.device("cpu")

print(f"Device: {DEVICE}")
print(f"PyTorch: {torch.__version__}")

# ─────────────────────────────────────────────
# DATA TRANSFORMS
# ─────────────────────────────────────────────
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

# ─────────────────────────────────────────────
# DATASETS
# ─────────────────────────────────────────────
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), train_transform)
val_dataset   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),   val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")
print(f"Classes detected: {train_dataset.classes}")

# ─────────────────────────────────────────────
# CLASS WEIGHTS
# ─────────────────────────────────────────────
class_counts = np.array([len(os.listdir(os.path.join(DATA_DIR, "train", c)))
                          for c in train_dataset.classes])
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float)
class_weights = class_weights / class_weights.sum() * NUM_CLASSES
print(f"Class weights: {dict(zip(train_dataset.classes, class_weights.numpy().round(3)))}")

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
model = models.resnet50(weights="IMAGENET1K_V1")

for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, NUM_CLASSES)
)

model = model.to(DEVICE)
print("Trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))

# ─────────────────────────────────────────────
# LOSS & OPTIMIZER
# ─────────────────────────────────────────────
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
optimizer = optim.Adam([
    {"params": model.layer3.parameters(), "lr": 1e-5},
    {"params": model.layer4.parameters(), "lr": 1e-5},
    {"params": model.fc.parameters(),     "lr": 1e-3}
])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# ─────────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────────
history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_acc   = 0.0
best_model = copy.deepcopy(model.state_dict())

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}  {'─'*40}")
    t0 = time.time()

    for phase in ["train", "val"]:
        model.train() if phase == "train" else model.eval()
        loader = train_loader if phase == "train" else val_loader

        running_loss, running_correct = 0.0, 0

        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                preds   = outputs.argmax(dim=1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss    += loss.item() * inputs.size(0)
            running_correct += (preds == labels).sum().item()

            if phase == "train" and batch_idx % 20 == 0:
                print(f"  Batch {batch_idx}/{len(loader)}  loss={loss.item():.4f}")

        epoch_loss = running_loss / len(loader.dataset)
        epoch_acc  = running_correct / len(loader.dataset)

        history[f"{phase}_loss"].append(epoch_loss)
        history[f"{phase}_acc"].append(epoch_acc)
        print(f"  {phase.upper():5s}  loss={epoch_loss:.4f}  acc={epoch_acc:.4f}")

        if phase == "val" and epoch_acc > best_acc:
            best_acc   = epoch_acc
            best_model = copy.deepcopy(model.state_dict())
            torch.save(best_model, os.path.join(MODEL_DIR, "best_resnet50_multiclass.pth"))
            print(f"  ✓ Best model saved  (val_acc={best_acc:.4f})")

    if phase == "train":
        scheduler.step()

    print(f"  Epoch time: {(time.time()-t0)/60:.1f} min")

# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────
print("\nFinal evaluation...")
model.load_state_dict(best_model)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs.to(DEVICE))
        preds   = outputs.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

report = classification_report(all_labels, all_preds,
                                target_names=train_dataset.classes, digits=4)
print("\nClassification Report:\n", report)

with open(os.path.join(RESULTS_DIR, "metrics", "classification_report.txt"), "w") as f:
    f.write(report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=train_dataset.classes,
            yticklabels=train_dataset.classes)
plt.title("Confusion Matrix — Multi-Class Damage Detection")
plt.ylabel("True Label"); plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "confusion_matrix_multiclass.png"), dpi=150)
plt.show()

# Training Curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(history["train_loss"], label="Train")
ax1.plot(history["val_loss"],   label="Val")
ax1.set_title("Loss"); ax1.legend()
ax2.plot(history["train_acc"], label="Train")
ax2.plot(history["val_acc"],   label="Val")
ax2.set_title("Accuracy"); ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "plots", "training_curves_multiclass.png"), dpi=150)
plt.show()

with open(os.path.join(RESULTS_DIR, "metrics", "history.json"), "w") as f:
    json.dump(history, f, indent=2)

print(f"\n✅ Stage 1 Complete. Best val accuracy: {best_acc:.4f}")
