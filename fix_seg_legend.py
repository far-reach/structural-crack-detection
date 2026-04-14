import torch, torch.nn as nn, numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib, os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

DATA_CORSPAL = "data/extra/spalling_corrosion_patches"
MODEL_DIR    = "models"
IMG_SIZE     = 256
NUM_CLASSES  = 3
CORROSION_COLOR = (255, 0, 0)
SPALLING_COLOR  = (255, 255, 0)
COLOR_MAP    = {0: (0,0,0), 1: (255,0,0), 2: (255,255,0)}
CLASS_NAMES  = ["Background", "Corrosion", "Spalling"]

model = deeplabv3_resnet50(weights="DEFAULT")
model.classifier[4]     = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.aux_classifier[4] = nn.Conv2d(256, NUM_CLASSES, kernel_size=1)
model.load_state_dict(torch.load(
    os.path.join(MODEL_DIR, "best_deeplabv3_segmentation.pth"),
    map_location="cpu"), strict=False)
model.eval()

tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        rgb[mask == cls] = color
    return rgb

src   = os.path.join(DATA_CORSPAL, "val")
files = sorted([f for f in os.listdir(src)
                if f.endswith(".png") and "_lab" not in f])

fig, axes = plt.subplots(3, 3, figsize=(12, 13))
fig.patch.set_facecolor("white")

with torch.no_grad():
    for i in range(3):
        name     = files[i*10]
        img      = Image.open(os.path.join(src, name)).convert("RGB").resize((IMG_SIZE,IMG_SIZE))
        mask_raw = np.array(Image.open(os.path.join(src,
                   name.replace(".png","_lab.png"))).convert("RGB").resize(
                   (IMG_SIZE,IMG_SIZE), Image.NEAREST))
        true_mask = np.zeros((IMG_SIZE,IMG_SIZE), dtype=np.int64)
        true_mask[np.all(mask_raw == CORROSION_COLOR, axis=2)] = 1
        true_mask[np.all(mask_raw == SPALLING_COLOR,  axis=2)] = 2

        tensor    = tf(img).unsqueeze(0)
        output    = model(tensor)["out"]
        pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()
        img_np    = np.array(img) / 255.0

        axes[i][0].imshow(img_np)
        axes[i][0].set_title("Input image", fontsize=11)
        axes[i][0].axis("off")

        axes[i][1].imshow(mask_to_rgb(true_mask))
        axes[i][1].set_title("Ground truth", fontsize=11)
        axes[i][1].axis("off")

        axes[i][2].imshow(mask_to_rgb(pred_mask))
        axes[i][2].set_title("Model prediction", fontsize=11)
        axes[i][2].axis("off")

# Add legend
patches = [mpatches.Patch(color=[c/255 for c in COLOR_MAP[i]],
           label=CLASS_NAMES[i]) for i in range(3)]
fig.legend(handles=patches, loc="lower center", ncol=3,
           fontsize=12, frameon=True, bbox_to_anchor=(0.5, 0.0))

plt.suptitle("Stage 2 — 3-Class Segmentation Results (mIoU = 0.789)",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig("results/plots/segmentation_results.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("Saved: results/plots/segmentation_results.png")