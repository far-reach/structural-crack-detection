"""
Per-class IoU for 4-class segmentation model
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os

DATA_CRACK   = "data/crack_seg"
DATA_CORSPAL = "data/extra/spalling_corrosion_patches"
MODEL_DIR    = "models"
IMG_SIZE     = 256
NUM_CLASSES  = 4
CORROSION_COLOR = (255, 0, 0)
SPALLING_COLOR  = (255, 255, 0)
CLASS_NAMES  = ["Background", "Crack", "Corrosion", "Spalling"]

class CrackDataset(Dataset):
    def __init__(self, split="val", img_size=256):
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
        mask  = np.array(Image.open(os.path.join(
                self.msk_dir, name)).convert("L").resize(
                (self.img_size, self.img_size), Image.NEAREST))
        label = np.zeros((self.img_size, self.img_size), dtype=np.int64)
        label[mask > 127] = 1
        return img, torch.tensor(label)

class CorSpalDataset(Dataset):
    def __init__(self, split="val", img_size=256):
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

model = deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "best_deeplabv3_4class.pth"),
    map_location="cpu"))
model.eval()

val_dataset = ConcatDataset([CrackDataset("val"), CorSpalDataset("val")])
loader      = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

intersection = np.zeros(NUM_CLASSES)
union        = np.zeros(NUM_CLASSES)
correct = total = 0

with torch.no_grad():
    for imgs, masks in loader:
        output = model(imgs)["out"]
        preds  = output.argmax(dim=1).numpy()
        masks  = masks.numpy()
        for cls in range(NUM_CLASSES):
            intersection[cls] += np.sum((preds == cls) & (masks == cls))
            union[cls]        += np.sum((preds == cls) | (masks == cls))
        correct += np.sum(preds == masks)
        total   += masks.size

iou = intersection / (union + 1e-10)
miou = np.mean(iou)
pixel_acc = correct / total

print("\n=== 4-CLASS SEGMENTATION EVALUATION ===")
print(f"{'Class':<15} {'IoU':>8}")
print("-" * 25)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<15} {iou[i]:>8.4f}")
print("-" * 25)
print(f"{'mIoU':<15} {miou:>8.4f}")
print(f"{'Pixel Acc':<15} {pixel_acc:>8.4f}")

with open("results/metrics/segmentation_4class_iou.txt", "w") as f:
    f.write("4-Class Segmentation IoU Breakdown\n")
    f.write("="*40 + "\n")
    for i, n in enumerate(CLASS_NAMES):
        f.write(f"{n}: {iou[i]:.4f}\n")
    f.write(f"mIoU: {miou:.4f}\n")
    f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")

print("\nSaved: results/metrics/segmentation_4class_iou.txt")