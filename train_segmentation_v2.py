"""
Stage 2 v2: 4-Class Semantic Segmentation
Classes: 0=Background, 1=Crack, 2=Corrosion, 3=Spalling
Author: Seyed Farhad Abtahi
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')

DATA_CRACK   = "data/crack_seg"
DATA_CORSPAL = "data/extra/spalling_corrosion_patches"
MODEL_DIR    = "models"
RESULTS_DIR  = "results"
NUM_CLASSES  = 4
BATCH_SIZE   = 4
NUM_EPOCHS   = 30
LR           = 2e-5
IMG_SIZE     = 256
DEVICE       = torch.device("cpu")

CORROSION_COLOR = (255, 0,   0)
SPALLING_COLOR  = (255, 255, 0)

print("Device : " + str(DEVICE))
print("PyTorch: " + str(torch.__version__))


class CrackDataset(Dataset):
    def __init__(self, split="train", img_size=256):
        self.img_size = img_size
        self.img_dir  = os.path.join(DATA_CRACK, split, "images")
        self.msk_dir  = os.path.join(DATA_CRACK, split, "masks")
        self.images   = sorted([f for f in os.listdir(self.img_dir)
                                if f.endswith(".png")])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name  = self.images[idx]
        img   = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        img   = img.resize((self.img_size, self.img_size))
        img   = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])(img)
        mask  = np.array(Image.open(os.path.join(self.msk_dir, name)).convert("L").resize(
                         (self.img_size, self.img_size), Image.NEAREST))
        label = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        label[mask > 127] = 1  # white = crack
        return img, torch.tensor(label)


class CorSpalDataset(Dataset):
    def __init__(self, split="train", img_size=256):
        self.img_size = img_size
        self.src_dir  = os.path.join(DATA_CORSPAL, split)
        all_files     = os.listdir(self.src_dir)
        self.images   = sorted([f for f in all_files
                                if f.endswith(".png") and "_lab" not in f])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name      = self.images[idx]
        mask_name = name.replace(".png", "_lab.png")
        img = Image.open(os.path.join(self.src_dir, name)).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])(img)
        mask  = np.array(Image.open(os.path.join(
                self.src_dir, mask_name)).convert("RGB").resize(
                (self.img_size, self.img_size), Image.NEAREST))
        label = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        label[np.all(mask == CORROSION_COLOR, axis=2)] = 2
        label[np.all(mask == SPALLING_COLOR,  axis=2)] = 3
        return img, torch.tensor(label)


# Combined datasets
train_dataset = ConcatDataset([
    CrackDataset("train", IMG_SIZE),
    CorSpalDataset("train", IMG_SIZE)
])
val_crack    = CrackDataset("val", IMG_SIZE)
val_corspal  = CorSpalDataset("val", IMG_SIZE)
val_dataset  = ConcatDataset([val_crack, val_corspal])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

print("Train: " + str(len(train_dataset)) + " | Val: " + str(len(val_dataset)))

model = deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model = model.to(DEVICE)

for name, param in model.named_parameters():
    if "layer3" in name or "layer4" in name or "classifier" in name or "aux_classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

print("Trainable params: " + str(sum(p.numel() for p in model.parameters()
      if p.requires_grad)))

# Heavy weight on crack class to handle extreme imbalance
class_weights = torch.tensor([0.1, 10.0, 1.0, 1.0])
NUM_EPOCHS = 30
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)


def compute_iou(preds, labels, num_classes=4):
    ious   = []
    preds  = preds.cpu().numpy().flatten()
    labels = labels.cpu().numpy().flatten()
    for cls in range(num_classes):
        intersection = np.sum((preds == cls) & (labels == cls))
        union        = np.sum((preds == cls) | (labels == cls))
        if union == 0:
            continue
        ious.append(intersection / union)
    return np.mean(ious)


history  = {"train_loss": [], "val_loss": [], "val_miou": []}
best_iou = 0.0

for epoch in range(NUM_EPOCHS):
    print("\nEpoch " + str(epoch+1) + "/" + str(NUM_EPOCHS))
    t0 = time.time()

    for phase in ["train", "val"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        loader       = train_loader if phase == "train" else val_loader
        running_loss = 0.0
        all_iou      = []

        for batch_idx, (imgs, masks) in enumerate(loader):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                output = model(imgs)["out"]
                loss   = criterion(output, masks)
                preds  = output.argmax(dim=1)
                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            all_iou.append(compute_iou(preds, masks))

            if phase == "train" and batch_idx % 50 == 0:
                print("  Batch " + str(batch_idx) + "/" + str(len(loader)) +
                      "  loss=" + str(round(loss.item(), 4)))

        epoch_loss = running_loss / len(loader.dataset)
        epoch_miou = np.mean(all_iou)

        history[phase + "_loss"].append(epoch_loss)
        if phase == "val":
            history["val_miou"].append(epoch_miou)

        print("  " + phase.upper() + "  loss=" + str(round(epoch_loss, 4)) +
              "  mIoU=" + str(round(epoch_miou, 4)))

        if phase == "val" and epoch_miou > best_iou:
            best_iou = epoch_miou
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, "best_deeplabv3_4class.pth"))
            print("  Best model saved (mIoU=" + str(round(best_iou, 4)) + ")")

    scheduler.step()
    print("  Epoch time: " + str(round((time.time()-t0)/60, 1)) + " min")

with open(os.path.join(RESULTS_DIR, "metrics",
          "segmentation_4class_history.json"), "w") as f:
    json.dump(history, f, indent=2)

print("Stage 2 v2 Complete. Best mIoU: " + str(round(best_iou, 4)))