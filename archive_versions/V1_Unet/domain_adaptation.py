"""
domain_adaptation.py — Unsupervised Test-Time Adaptation (GRAM TTA)

This script implements the core idea of GRAM (Generalized Region-Aware Mixture-of-Experts)
without requiring a massive multi-expert architecture.
It uses Unsupervised Pseudo-Labeling with Confidence Filtering to adapt to a new city at test-time.
"""

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config
from train import build_model, get_train_transforms, compute_metrics, CombinedLoss

CONFIDENCE_THRESHOLD = 0.90
UNCERTAINTY_TOLERANCE = 0.10  # Max percentage of uncertain pixels allowed in a tile
TTA_EPOCHS = 3
TTA_LR = 1e-5


class TTADataset(Dataset):
    """Dataset for training on in-memory images and pseudo-labels."""
    def __init__(self, images, pseudo_labels, transform=None):
        self.images = images
        self.pseudo_labels = pseudo_labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        mask = self.pseudo_labels[idx]

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented["image"]
            mask = augmented["mask"]

        # Ensure correct shapes
        mask = mask.unsqueeze(0) if mask.ndim == 2 else mask
        return img, mask


@torch.no_grad()
def generate_pseudo_labels(model, test_images_dir, device):
    """
    Runs the model on unseen test images, filters out noisy predictions,
    and returns high-confidence (image, pseudo-label) pairs.
    """
    model.eval()
    
    image_paths = glob.glob(os.path.join(test_images_dir, "*.npy"))
    print(f"[TTA] Found {len(image_paths)} unlabeled test tiles in {test_images_dir}.")

    confident_images = []
    confident_labels = []

    # Simple transform to tensor
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    eval_transform = A.Compose([
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2()
    ])

    pbar = tqdm(image_paths, desc="Generating Pseudo-Labels", unit="tile")
    for img_path in pbar:
        img_np = np.load(img_path)  # (H, W, 3) uint8
        
        # Prepare for model
        aug = eval_transform(image=img_np)
        img_tensor = aug["image"].unsqueeze(0).to(device)  # (1, 3, H, W)

        # Predict
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze()  # (H, W)

        # Confidence Filtering (The GRAM magic)
        # We consider a pixel "certain" if prob > CONFIDENCE_THRESHOLD or prob < (1 - CONFIDENCE_THRESHOLD)
        certain_pixels = (probs > CONFIDENCE_THRESHOLD) | (probs < (1 - CONFIDENCE_THRESHOLD))
        percent_certain = certain_pixels.float().mean().item()

        # Only keep this tile if the model is very sure about MOST of the pixels
        if percent_certain >= (1.0 - UNCERTAINTY_TOLERANCE):
            pseudo_mask = (probs > 0.5).float().cpu().numpy()
            confident_images.append(img_np)
            confident_labels.append(pseudo_mask)

    print(f"\n[TTA] Kept {len(confident_images)} / {len(image_paths)} tiles with high confidence.")
    return confident_images, confident_labels


def adapt_model():
    print("=" * 70)
    print("  Slum Segmentation — Unsupervised Test-Time Adaptation (TTA)")
    print("  Inspired by GRAM (AAAI 2026)")
    print("=" * 70)

    device = config.DEVICE

    # 1. Load the Universal Model
    model = build_model()
    if not os.path.exists(config.MODEL_SAVE_PATH):
        raise FileNotFoundError(f"Model not found at {config.MODEL_SAVE_PATH}")
    
    print("[INFO] Loading Universal Model...")
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 2. Generate Pseudo-Labels on the Test Domain (Casablanca)
    test_images_dir = os.path.join(config.TEST_DIR, "images")
    images, pseudo_labels = generate_pseudo_labels(model, test_images_dir, device)

    if len(images) == 0:
        print("[WARNING] No confident pseudo-labels generated. Adaptation aborted.")
        return

    # 3. Prepare Dataset for Adaptation
    tta_dataset = TTADataset(images, pseudo_labels, transform=get_train_transforms())
    tta_loader = DataLoader(tta_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)

    # 4. Fine-Tune on Pseudo-Labels (Unsupervised)
    print("\n[INFO] Starting Unsupervised Fine-Tuning on Pseudo-Labels...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=TTA_LR, weight_decay=config.WEIGHT_DECAY)
    criterion = CombinedLoss(bce_weight=config.BCE_WEIGHT, dice_weight=config.DICE_WEIGHT)

    model.train()
    for epoch in range(1, TTA_EPOCHS + 1):
        running_loss = 0.0
        pbar = tqdm(tta_loader, desc=f"  Epoch {epoch}/{TTA_EPOCHS}", unit="batch")
        
        for imgs, masks in pbar:
            imgs = imgs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})

        print(f"Epoch {epoch} | Loss: {running_loss / len(tta_loader):.4f}")

    # 5. Save the adapted model
    adapted_path = config.MODEL_SAVE_PATH.replace(".pth", "_adapted.pth")
    save_dict = {
        "model_state_dict": model.state_dict(),
        "config": checkpoint.get("config", {})
    }
    torch.save(save_dict, adapted_path)
    print(f"\n[DONE] Adapted model saved to {adapted_path}")
    print("=" * 70)


if __name__ == "__main__":
    adapt_model()
