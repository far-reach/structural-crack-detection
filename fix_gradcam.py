import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torchvision import models, transforms
from PIL import Image
import os
import cv2

DEVICE     = torch.device("cpu")
MODEL_PATH = "models/best_resnet50_multiclass.pth"
DATA_DIR   = "data/codebrim/val"
CLASSES    = ["crack", "intact", "corrosion", "spalling"]

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(model.fc.in_features, 256),
    nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

gradients  = []
activations = []

def save_gradient(grad):
    gradients.append(grad)

def forward_hook(module, input, output):
    activations.append(output)
    output.register_hook(save_gradient)

handle = model.layer4[-1].register_forward_hook(forward_hook)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def get_gradcam(img_path):
    gradients.clear()
    activations.clear()
    img    = Image.open(img_path).convert("RGB")
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
    img_np  = np.array(img.resize((224, 224)))
    img_f   = img_np / 255.0
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.5 * img_f + 0.5 * heatmap
    return img_np, cam, overlay, CLASSES[pred], round(conf*100, 1)

# Find correct prediction for each class
# Search more files and use ANY file if no correct found
found = {}
for cls in CLASSES:
    cls_dir = os.path.join(DATA_DIR, cls)
    if not os.path.exists(cls_dir):
        print(f"WARNING: {cls_dir} not found")
        continue
    files = [f for f in os.listdir(cls_dir)
             if f.endswith(('.jpg','.png','.jpeg'))]
    print(f"Searching {len(files)} files for correct {cls} prediction...")
    best = None
    for fname in files:
        path = os.path.join(cls_dir, fname)
        try:
            result = get_gradcam(path)
            if result[3] == cls:  # correct prediction
                found[cls] = (path,) + result
                print(f"  Found correct: {fname} conf={result[4]}%")
                break
            elif best is None:
                best = (path,) + result  # keep as fallback
        except Exception as e:
            continue
    if cls not in found:
        if best is not None:
            found[cls] = best
            print(f"  Using best fallback for {cls}: conf={best[4]}%")
        else:
            print(f"  ERROR: No valid image found for {cls}")

print(f"\nFound examples for: {list(found.keys())}")

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.patch.set_facecolor('white')

for row, cls in enumerate(CLASSES):
    if cls not in found:
        for col in range(3):
            axes[row][col].text(0.5, 0.5, f'No image\nfound for\n{cls}',
                               ha='center', va='center', transform=axes[row][col].transAxes)
            axes[row][col].axis("off")
        continue

    path, img_np, cam, overlay, pred_cls, conf = found[cls]
    correct = pred_cls == cls
    color   = "green" if correct else "red"
    status  = "✓ Correct" if correct else "✗ Wrong"

    axes[row][0].imshow(img_np)
    axes[row][0].set_title(f"Input: {cls.upper()}", fontsize=11, fontweight='bold')
    axes[row][0].axis("off")

    axes[row][1].imshow(cam, cmap="jet")
    axes[row][1].set_title("GradCAM heatmap", fontsize=11)
    axes[row][1].axis("off")

    axes[row][2].imshow(np.clip(overlay, 0, 1))
    axes[row][2].set_title(
        f"Pred: {pred_cls.upper()} ({conf}%)\n{status}",
        fontsize=11, color=color
    )
    axes[row][2].axis("off")

plt.suptitle(
    "GradCAM — Model Attention Maps for Structural Damage Classes\n"
    "Red regions = high attention; confirms damage-specific feature localization",
    fontsize=12, fontweight="bold", y=1.01
)
plt.tight_layout()
plt.savefig("results/plots/gradcam_visualization.png",
            dpi=150, bbox_inches="tight", facecolor="white")
print("\nSaved: results/plots/gradcam_visualization.png")
handle.remove()