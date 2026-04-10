"""
Structural Crack Detection - Dataset & Data Loading
Compatible with Kaggle Concrete Crack Images dataset:
https://www.kaggle.com/datasets/arunrk7/surface-crack-detection

Dataset structure expected:
    data/
        Positive/   <- cracked images
        Negative/   <- intact images
"""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ImageNet normalization (standard for ResNet transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASSES = {0: "Negative (No Crack)", 1: "Positive (Crack Detected)"}
CLASS_COLORS = {0: "#2ecc71", 1: "#e74c3c"}


def get_transforms(mode="train"):
    """
    Returns image transforms for training or inference.
    Training: augmentation (flip, rotation, color jitter)
    Val/Test: deterministic resize + normalize only
    """
    if mode == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])


class CrackDataset(Dataset):
    """
    Custom dataset for concrete crack images.
    
    Expects:
        root_dir/Positive/  <- cracked images (label=1)
        root_dir/Negative/  <- intact images  (label=0)
    """

    def __init__(self, root_dir, mode="train", val_split=0.2, seed=42):
        self.root_dir = Path(root_dir)
        self.transform = get_transforms(mode)
        self.mode = mode

        # Collect all image paths and labels
        self.samples = []

        for label, folder in [(1, "Positive"), (0, "Negative")]:
            folder_path = self.root_dir / folder
            if not folder_path.exists():
                raise FileNotFoundError(f"Folder not found: {folder_path}")
            
            images = sorted([
                (str(p), label) 
                for p in folder_path.glob("*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]
            ])
            self.samples.extend(images)

        # Train/val split
        import random
        random.seed(seed)
        random.shuffle(self.samples)

        n_val = int(len(self.samples) * val_split)
        if mode == "train":
            self.samples = self.samples[n_val:]
        elif mode == "val":
            self.samples = self.samples[:n_val]
        # mode == "all": keep everything

        print(f"[{mode.upper()}] Loaded {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_dataloaders(data_dir, batch_size=32, num_workers=4):
    """Returns train and val DataLoaders."""
    train_dataset = CrackDataset(data_dir, mode="train")
    val_dataset   = CrackDataset(data_dir, mode="val")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test with a single image
    import sys
    if len(sys.argv) > 1:
        dataset = CrackDataset(sys.argv[1], mode="all")
        print(f"Total samples: {len(dataset)}")
        img, label = dataset[0]
        print(f"Image tensor shape: {img.shape}")
        print(f"Label: {label} -> {CLASSES[label]}")
