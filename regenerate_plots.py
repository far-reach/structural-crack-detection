import matplotlib
matplotlib.use('Agg')
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import os

DEVICE    = torch.device("cpu")
DATA_DIR  = "data/codebrim"
MODEL_PATH = "models/best_resnet50_multiclass.pth"

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), val_transform)
val_loader  = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(model.fc.in_features, 256),
    nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        preds   = outputs.argmax(dim=1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

classes = val_dataset.classes
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes)
plt.title("Confusion Matrix - Multi-Class Damage Detection")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()
plt.savefig("results/plots/confusion_matrix_multiclass.png", dpi=150)
plt.savefig("results/plots/confusion_matrix_multiclass.jpg", dpi=150)
print("Confusion matrix saved.")

report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
print(report)