"""
Visualization for 4-class segmentation model
Author: Seyed Farhad Abtahi
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
matplotlib.use('Agg')
import os

DATA_CRACK   = "data/crack_seg"
DATA_CORSPAL = "data/extra/spalling_corrosion_patches"
MODEL_DIR    = "models"
IMG_SIZE     = 256
NUM_CLASSES  = 4
CORROSION_COLOR = (255, 0, 0)
SPALLING_COLOR  = (255, 255, 0)

COLOR_MAP   = {0: (0,0,0), 1: (255,0,0), 2: (255,165,0), 3: (255,255,0)}
CLASS_NAMES = ["Background", "Crack", "Corrosion", "Spalling"]

model = deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "best_deeplabv3_4class.pth"),
    map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb

def get_crack_sample(idx=5):
    img_dir = os.path.join(DATA_CRACK, "val", "images")
    msk_dir = os.path.join(DATA_CRACK, "val", "masks")
    files   = sorted(os.listdir(img_dir))
    name    = files[idx]
    img     = Image.open(os.path.join(img_dir, name)).convert("RGB")
    img     = img.resize((IMG_SIZE, IMG_SIZE))
    mask    = np.array(Image.open(os.path.join(msk_dir, name)).convert("L").resize(
                       (IMG_SIZE, IMG_SIZE), Image.NEAREST))
    label   = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
    label[mask > 127] = 1
    tensor  = transform(img)
    return np.array(img)/255.0, label, tensor

def get_corspal_sample(idx=10):
    src   = os.path.join(DATA_CORSPAL, "val")
    files = sorted([f for f in os.listdir(src)
                    if f.endswith(".png") and "_lab" not in f])
    name  = files[idx]
    img   = Image.open(os.path.join(src, name)).convert("RGB")
    img   = img.resize((IMG_SIZE, IMG_SIZE))
    mask  = np.array(Image.open(os.path.join(src,
                     name.replace(".png","_lab.png"))).convert("RGB").resize(
                     (IMG_SIZE, IMG_SIZE), Image.NEAREST))
    label = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int64)
    label[np.all(mask == CORROSION_COLOR, axis=2)] = 2
    label[np.all(mask == SPALLING_COLOR,  axis=2)] = 3
    tensor = transform(img)
    return np.array(img)/255.0, label, tensor

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.patch.set_facecolor("white")

samples = [
    get_crack_sample(5),
    get_crack_sample(15),
    get_corspal_sample(10),
]
titles  = ["UAV Crack Sample 1", "UAV Crack Sample 2", "Corrosion/Spalling Sample"]

with torch.no_grad():
    for i, (img_np, true_mask, tensor) in enumerate(samples):
        output    = model(tensor.unsqueeze(0))["out"]
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

        axes[i][0].imshow(img_np)
        axes[i][0].set_title(f"Input: {titles[i]}", fontsize=10)
        axes[i][0].axis("off")

        axes[i][1].imshow(mask_to_rgb(true_mask))
        axes[i][1].set_title("Ground Truth", fontsize=10)
        axes[i][1].axis("off")

        axes[i][2].imshow(mask_to_rgb(pred_mask))
        axes[i][2].set_title("Prediction", fontsize=10)
        axes[i][2].axis("off")

patches = [mpatches.Patch(color=[c/255 for c in COLOR_MAP[i]],
           label=CLASS_NAMES[i]) for i in range(4)]
fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=10)
plt.suptitle("DeepLabV3+ 4-Class Segmentation — Structural Damage",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/plots/segmentation_4class_results.png",
            dpi=150, bbox_inches="tight")
print("Saved: results/plots/segmentation_4class_results.png")