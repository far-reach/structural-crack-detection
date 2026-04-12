import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

DATA_DIR  = "data/extra/spalling_corrosion_patches"
MODEL_DIR = "models"
IMG_SIZE  = 256
NUM_CLASSES = 3
CORROSION_COLOR = (255, 0, 0)
SPALLING_COLOR  = (255, 255, 0)
COLOR_MAP   = {0: (0,0,0), 1: (255,0,0), 2: (255,255,0)}
CLASS_NAMES = ["Background", "Corrosion", "Spalling"]

class SegmentationDataset(Dataset):
    def __init__(self, split="val", img_size=256):
        self.img_size = img_size
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
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])(img)
        label = self.mask_to_label(os.path.join(self.src_dir, mask_name))
        return img, label

model = deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR,
                      "best_deeplabv3_segmentation.pth")), strict=False)
model.eval()

dataset = SegmentationDataset("val", IMG_SIZE)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb

with torch.no_grad():
    for i in range(3):
        img_tensor, true_mask = dataset[i * 10]
        output    = model(img_tensor.unsqueeze(0))["out"]
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
        img_show  = img_tensor.numpy().transpose(1,2,0)
        img_show  = (img_show * [0.229,0.224,0.225] +
                     [0.485,0.456,0.406]).clip(0,1)
        axes[i][0].imshow(img_show);      axes[i][0].set_title("Input");       axes[i][0].axis("off")
        axes[i][1].imshow(mask_to_rgb(true_mask.numpy())); axes[i][1].set_title("Ground Truth"); axes[i][1].axis("off")
        axes[i][2].imshow(mask_to_rgb(pred_mask));         axes[i][2].set_title("Prediction");   axes[i][2].axis("off")

patches = [mpatches.Patch(color=[c/255 for c in COLOR_MAP[i]],
           label=CLASS_NAMES[i]) for i in range(3)]
fig.legend(handles=patches, loc="lower center", ncol=3)
plt.suptitle("DeepLabV3+ Segmentation — Structural Damage", fontsize=14)
plt.tight_layout()
plt.savefig("results/plots/segmentation_results.png", dpi=150)
print("Saved: results/plots/segmentation_results.png")