"""
GradCAM Visualization for Stage 1 Classifier
Shows which regions the model focuses on for each damage class
Author: Seyed Farhad Abtahi
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torchvision import models, transforms, datasets
from PIL import Image
import os
import cv2

DEVICE     = torch.device("cpu")
MODEL_PATH = "models/best_resnet50_multiclass.pth"
DATA_DIR   = "data/codebrim/val"
RESULTS    = "results/plots"
CLASSES    = ["crack", "intact", "corrosion", "spalling"]

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(model.fc.in_features, 256),
    nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# GradCAM hooks
gradients = []
activations = []

def save_gradient(grad):
    gradients.append(grad)

def forward_hook(module, input, output):
    activations.append(output)
    output.register_hook(save_gradient)

# Register hook on last conv layer
handle = model.layer4[-1].register_forward_hook(forward_hook)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_gradcam(img_path):
    gradients.clear()
    activations.clear()

    img = Image.open(img_path).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    output = model(tensor)
    pred   = output.argmax(dim=1).item()
    conf   = torch.softmax(output, dim=1).max().item()

    model.zero_grad()
    output[0, pred].backward()

    grad = gradients[0].squeeze().numpy()
    act  = activations[0].squeeze().detach().numpy()

    weights = grad.mean(axis=(1, 2))
    cam = np.zeros(act.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * act[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()

    img_resized = np.array(img.resize((224, 224))) / 255.0
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.5 * img_resized + 0.5 * heatmap

    return img_resized, cam, overlay, CLASSES[pred], round(conf * 100, 1)

# Get one sample per class
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.patch.set_facecolor("white")

for row, cls in enumerate(CLASSES):
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.exists(cls_dir):
        continue
    files = [f for f in os.listdir(cls_dir) if f.endswith(('.jpg','.png','.jpeg'))]
    if not files:
        continue
    img_path = os.path.join(cls_dir, files[5])

    img_orig, cam, overlay, pred, conf = get_gradcam(img_path)

    axes[row][0].imshow(img_orig)
    axes[row][0].set_title(f"Input: {cls.upper()}", fontsize=11)
    axes[row][0].axis("off")

    axes[row][1].imshow(cam, cmap="jet")
    axes[row][1].set_title("GradCAM Heatmap", fontsize=11)
    axes[row][1].axis("off")

    axes[row][2].imshow(overlay)
    axes[row][2].set_title(f"Overlay — Pred: {pred.upper()} {conf}%", fontsize=11)
    axes[row][2].axis("off")

plt.suptitle("GradCAM — Model Attention Maps for Structural Damage Classes",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS, "gradcam_visualization.png"),
            dpi=150, bbox_inches="tight")
print("Saved: results/plots/gradcam_visualization.png")

handle.remove()