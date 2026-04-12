"""
Stage 2: Semantic Segmentation for Structural Damage
DeepLabV3+ with ResNet50 backbone
Classes: 0=background, 1=corrosion, 2=spalling
Author: Seyed Farhad Abtahi
"""

import os
import time
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_DIR    = "data/extra/spalling_corrosion_patches"
MODEL_DIR   = "models"
RESULTS_DIR = "results"
NUM_CLASSES = 3
BATCH_SIZE  = 4
NUM_EPOCHS  = 20
LR          = 5e-5
IMG_SIZE    = 256
DEVICE      = torch.device("cpu")

CORROSION_COLOR = (255, 0, 0)
SPALLING_COLOR  = (255, 255, 0)

print("Device : " + str(DEVICE))
print("PyTorch: " + str(torch.__version__))


class SegmentationDataset(Dataset):
    def __init__(self, split="train", img_size=256):
        self.img_size = img_size
        self.split    = split
        self.src_dir  = os.path.join(DATA_DIR, split)
        all_files     = os.listdir(self.src_dir)
        self.images   = sorted([f for f in all_files
                                if f.endswith(".png") and "_lab" not in f])

    def __len__(self):
        return len(self.images)

    def mask_to_label(self, mask_path):
        mask  = np.array(Image.open(mask_path).convert("RGB").resize(
                         (self.img_size, self.img_size), Image.NEAREST))
        label = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        label[np.all(mask == CORROSION_COLOR, axis=2)] = 1
        label[np.all(mask == SPALLING_COLOR,  axis=2)] = 2
        return torch.tensor(label)

    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace(".png", "_lab.png")
        img = Image.open(os.path.join(self.src_dir, img_name)).convert("RGB")
        img = img.resize((self.img_size, self.img_size))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])(img)
        mask_path = os.path.join(self.src_dir, mask_name)
        label     = self.mask_to_label(mask_path)
        return img, label


train_dataset = SegmentationDataset("train", IMG_SIZE)
val_dataset   = SegmentationDataset("val",   IMG_SIZE)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader    = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("Train: " + str(len(train_dataset)) + " | Val: " + str(len(val_dataset)))

model = deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model = model.to(DEVICE)

for name, param in model.named_parameters():
    if "layer4" in name or "classifier" in name or "aux_classifier" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

print("Trainable params: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.5)


def compute_iou(preds, labels, num_classes=3):
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
                print("  Batch " + str(batch_idx) + "/" + str(len(loader)) + "  loss=" + str(round(loss.item(), 4)))

        epoch_loss = running_loss / len(loader.dataset)
        epoch_miou = np.mean(all_iou)

        history[phase + "_loss"].append(epoch_loss)
        if phase == "val":
            history["val_miou"].append(epoch_miou)

        print("  " + phase.upper() + "  loss=" + str(round(epoch_loss, 4)) + "  mIoU=" + str(round(epoch_miou, 4)))

        if phase == "val" and epoch_miou > best_iou:
            best_iou = epoch_miou
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_deeplabv3_segmentation.pth"))
            print("  Best model saved (mIoU=" + str(round(best_iou, 4)) + ")")

    scheduler.step()
    print("  Epoch time: " + str(round((time.time()-t0)/60, 1)) + " min")


print("\nGenerating visualizations...")
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_deeplabv3_segmentation.pth")))
model.eval()

COLOR_MAP   = {0: (0,0,0), 1: (255,0,0), 2: (255,255,0)}
CLASS_NAMES = ["Background", "Corrosion", "Spalling"]

fig, axes      = plt.subplots(3, 3, figsize=(12, 12))
sample_dataset = SegmentationDataset("val", IMG_SIZE)

with torch.no_grad():
    for i in range(3):
        img_tensor, true_mask = sample_dataset[i * 10]
        output    = model(img_tensor.unsqueeze(0).to(DEVICE))["out"]
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

        img_show = img_tensor.numpy().transpose(1, 2, 0)
        img_show = (img_show * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

        def mask_to_rgb(mask):
            rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
            for cls, color in COLOR_MAP.items():
                rgb[mask == cls] = color
            return rgb

        axes[i][0].imshow(img_show)
        axes[i][0].set_title("Input Image")
        axes[i][0].axis("off")
        axes[i][1].imshow(mask_to_rgb(true_mask.numpy()))
        axes[i][1].set_title("Ground Truth")
        axes[i][1].axis("off")
        axes[i][2].imshow(mask_to_rgb(pred_mask))
        axes[i][2]