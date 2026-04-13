"""
Inference Time Benchmarking
Reports CPU inference time per frame for UAV deployment planning
Author: Seyed Farhad Abtahi
"""

import torch
import torch.nn as nn
import time
import numpy as np
from torchvision import models, transforms
from PIL import Image
import os

DEVICE     = torch.device("cpu")
MODEL_PATH = "models/best_resnet50_multiclass.pth"
DATA_DIR   = "data/codebrim/val/crack"
NUM_RUNS   = 50

model = models.resnet50(weights=None)
model.fc = nn.Sequential(
    nn.Dropout(0.4), nn.Linear(model.fc.in_features, 256),
    nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 4)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.jpg','.png'))][:NUM_RUNS]
times = []

print(f"Benchmarking {NUM_RUNS} frames on CPU...")

for f in files:
    img = Image.open(os.path.join(DATA_DIR, f)).convert("RGB")
    tensor = transform(img).unsqueeze(0)

    start = time.perf_counter()
    with torch.no_grad():
        output = model(tensor)
    end = time.perf_counter()
    times.append((end - start) * 1000)

mean_ms  = round(np.mean(times), 2)
std_ms   = round(np.std(times), 2)
fps      = round(1000 / mean_ms, 2)
gpu_est  = round(mean_ms / 15, 2)

print(f"\n=== INFERENCE TIME REPORT ===")
print(f"Device          : CPU (Intel/AMD)")
print(f"Mean time/frame : {mean_ms} ms")
print(f"Std deviation   : {std_ms} ms")
print(f"CPU throughput  : {fps} FPS")
print(f"Est. GPU time   : ~{gpu_est} ms/frame (15x speedup estimate)")
print(f"Est. GPU FPS    : ~{round(1000/gpu_est)} FPS")
print(f"\nFor a 4x5 UAV grid (20 frames):")
print(f"  CPU total : {round(mean_ms*20/1000, 2)} seconds")
print(f"  GPU total : ~{round(gpu_est*20/1000, 2)} seconds")

with open("results/metrics/inference_time.txt", "w") as f:
    f.write(f"Mean CPU inference time: {mean_ms} ms/frame\n")
    f.write(f"CPU throughput: {fps} FPS\n")
    f.write(f"Estimated GPU time: {gpu_est} ms/frame\n")
    f.write(f"Estimated GPU FPS: {round(1000/gpu_est)} FPS\n")

print("\nSaved: results/metrics/inference_time.txt")