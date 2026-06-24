"""
finetune.py — Expert Few-Shot Fine-Tuning.
Loads the Universal best_model.pth, applies a very low learning rate,
and trains it only on Casablanca for a few epochs to correct False Positives.
"""

import os
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from train import SlumDataset, get_train_transforms, get_val_transforms, build_model, compute_metrics, CombinedLoss

# Hyperparameters for Fine-Tuning
FINETUNE_LR = 1e-5
FINETUNE_EPOCHS = 10


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_metrics = {"iou": 0.0, "dice": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        batch_metrics = compute_metrics(logits, masks)
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k]
        
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "F1": f"{batch_metrics['f1']:.4f}"})

    n = len(loader)
    return running_loss / n, {k: v / n for k, v in running_metrics.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_metrics = {"iou": 0.0, "dice": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item()
        batch_metrics = compute_metrics(logits, masks)
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k]

    n = len(loader)
    return running_loss / n, {k: v / n for k, v in running_metrics.items()}


def main():
    print("=" * 70)
    print("  Slum Segmentation — EXPERT Fine-Tuning")
    print(f"  Target: Casablanca | LR: {FINETUNE_LR} | Epochs: {FINETUNE_EPOCHS}")
    print("=" * 70)

    device = config.DEVICE

    # 1. Load Data
    # For fine-tuning, we use the training set but ideally just the new city.
    # We will filter out tiles that are not from Casablanca if we want pure fine-tuning.
    # To keep it simple, we train on the whole train_loader but with the new Casablanca tiles included.
    # Actually, it's better to train on everything to prevent Catastrophic Forgetting.
    print(f"[INFO] Loading data from {config.TRAIN_DIR}")
    train_ds = SlumDataset(config.TRAIN_DIR, transform=get_train_transforms())
    val_ds = SlumDataset(config.VAL_DIR, transform=get_val_transforms())

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    # 2. Load the Pre-Trained "Universal" Model
    model = build_model()
    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"[❌ ERROR] Pre-trained model not found at {config.MODEL_SAVE_PATH}")
        return
        
    print(f"[INFO] Loading existing universal weights...")
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 3. Optimiser & Scheduler (Very Low LR)
    criterion = CombinedLoss(bce_weight=config.BCE_WEIGHT, dice_weight=config.DICE_WEIGHT)
    optimizer = torch.optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=config.WEIGHT_DECAY)
    
    # 4. Fine-Tuning Loop
    best_val_loss = float("inf")
    
    print("\n" + "─" * 70)
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>11}  "
          f"{'IoU':>7}  {'F1':>7}  {'PR':>7}  {'RC':>7}  Status")
    print("─" * 70)

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metrics = validate(model, val_loader, criterion, device)

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            status = "★ Best"
            save_dict = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "model_arch": config.MODEL_ARCH,
                    "unet_encoder": config.UNET_ENCODER if hasattr(config, "UNET_ENCODER") else None,
                    "in_channels": config.IN_CHANNELS,
                    "num_classes": config.NUM_CLASSES,
                }
            }
            torch.save(save_dict, config.MODEL_SAVE_PATH)

        print(
            f"{epoch:>6d}  {train_loss:>11.4f}  {val_loss:>11.4f}  "
            f"{val_metrics['iou']:>7.4f}  {val_metrics['f1']:>7.4f}  "
            f"{val_metrics['precision']:>7.4f}  {val_metrics['recall']:>7.4f}  {status}"
        )

    print("\n[DONE] Fine-Tuning complete. The model is now adapted to Moroccan architecture.")


if __name__ == "__main__":
    main()
