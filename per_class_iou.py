"""
Per-class IoU breakdown for segmentation model
Author: Seyed Farhad Abtahi
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import Dataset, DataLoader
import os

DATA_DIR  = "data/extra/spalling_corrosion_patches"
MODEL_DIR = "models"
IMG_SIZE  = 256
NUM_CLASSES = 3
CORROSION_COLOR = (255, 0, 0)
SPALLING_COLOR  = (255, 255, 0)
CLASS_NAMES = ["Background", "Corrosion", "Spalling"]

class SegDataset(Dataset):
    def __init__(self, split="val"):
        self.src = os.path.join(DATA_DIR, split)
        all_files = os.listdir(self.src)
        self.images = sorted([f for f in all_files
                              if f.endswith(".png") and "_lab" not in f])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_name  = self.images[idx]
        mask_name = img_name.replace(".png", "_lab.png")
        img = Image.open(os.path.join(self.src, img_name)).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])(img)
        mask = np.array(Image.open(os.path.join(self.src, mask_name)).convert("RGB").resize(
                        (IMG_SIZE, IMG_SIZE), Image.NEAREST))
        label = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
        label[np.all(mask == CORROSION_COLOR, axis=2)] = 1
        label[np.all(mask == SPALLING_COLOR,  axis=2)] = 2
        return img, torch.tensor(label)

model = deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR,
                      "best_deeplabv3_segmentation.pth"),
                      map_location="cpu"), strict=False)
model.eval()

dataset = SegDataset("val")
loader  = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)

intersection = np.zeros(NUM_CLASSES)
union        = np.zeros(NUM_CLASSES)
correct      = 0
total        = 0

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

iou_per_class = intersection / (union + 1e-10)
miou          = np.mean(iou_per_class)
pixel_acc     = correct / total

print("\n=== SEGMENTATION EVALUATION ===")
print(f"{'Class':<15} {'IoU':>8} {'Intersection':>15} {'Union':>10}")
print("-" * 52)
for i, name in enumerate(CLASS_NAMES):
    print(f"{name:<15} {iou_per_class[i]:>8.4f} {int(intersection[i]):>15,} {int(union[i]):>10,}")
print("-" * 52)
print(f"{'mIoU':<15} {miou:>8.4f}")
print(f"{'Pixel Acc':<15} {pixel_acc:>8.4f}")

with open("results/metrics/segmentation_iou.txt", "w") as f:
    f.write("Per-Class IoU Breakdown\n")
    f.write("="*40 + "\n")
    for i, name in enumerate(CLASS_NAMES):
        f.write(f"{name}: {iou_per_class[i]:.4f}\n")
    f.write(f"mIoU: {miou:.4f}\n")
    f.write(f"Pixel Accuracy: {pixel_acc:.4f}\n")

print("\nSaved: results/metrics/segmentation_iou.txt")