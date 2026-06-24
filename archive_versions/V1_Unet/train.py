# train.py — Model definition, Dataset class, and full training loop.

import os
import glob
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

import config


# ════════════════════════════════════════════════════════════════
#  1. DATASET
# ════════════════════════════════════════════════════════════════

class SlumDataset(Dataset):
    def __init__(self, split_dir: str, transform=None):
        self.image_paths = sorted(glob.glob(os.path.join(split_dir, "images", "*.npy")))
        self.mask_paths = sorted(glob.glob(os.path.join(split_dir, "masks", "*.npy")))

        assert len(self.image_paths) == len(self.mask_paths), (
            f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks "
            f"in {split_dir}"
        )
        assert len(self.image_paths) > 0, f"No tiles found in {split_dir}"

        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.load(self.image_paths[idx])   
        mask = np.load(self.mask_paths[idx])      

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]            
            mask = augmented["mask"]              
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            mask = torch.from_numpy(mask)

        mask = mask.float().unsqueeze(0)
        return image, mask


# ────────────────────────────────────────────────────────────────
#  Augmentation pipelines
# ────────────────────────────────────────────────────────────────

def get_train_transforms():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        # Higher scale variation to handle 40cm vs 50cm vs Xcm
        A.Affine(translate_percent=0.1, scale=(0.8, 1.2), rotate=(-45, 45), p=0.7),
        # Strong color jitter to handle different soils, roofs, and sensor colors
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.GridDistortion(p=0.3),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])

def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


# ════════════════════════════════════════════════════════════════
#  2. MODEL
# ════════════════════════════════════════════════════════════════

def build_model() -> nn.Module:
    if config.MODEL_ARCH == "segformer":
        print("[MODEL] SegFormer-B0  →  FPN(encoder=mit_b0)")
        model = smp.FPN(
            encoder_name="mit_b0",
            encoder_weights="imagenet",
            in_channels=config.IN_CHANNELS,
            classes=config.NUM_CLASSES,
        )
    elif config.MODEL_ARCH == "unet":
        print(f"[MODEL] U-Net  →  encoder={config.UNET_ENCODER}")
        model = smp.Unet(
            encoder_name=config.UNET_ENCODER,
            encoder_weights="imagenet",
            in_channels=config.IN_CHANNELS,
            classes=config.NUM_CLASSES,
        )
    else:
        raise ValueError(f"Unknown MODEL_ARCH: {config.MODEL_ARCH}. Use 'segformer' or 'unet'.")

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[MODEL] Parameters: {total_params:,} total  |  {trainable:,} trainable")

    return model


# ════════════════════════════════════════════════════════════════
#  3. LOSS FUNCTION
# ════════════════════════════════════════════════════════════════

class CombinedLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
        super().__init__()
        self.focal_weight = bce_weight
        self.dice_weight = dice_weight
        from torchvision.ops import sigmoid_focal_loss
        self.focal_fn = sigmoid_focal_loss
        self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # CORRECTION MAJEURE ICI : alpha=0.8 pour forcer le modèle à regarder les bidonvilles
        focal_loss = self.focal_fn(logits, targets, alpha=0.8, gamma=2.0, reduction='mean')
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# ════════════════════════════════════════════════════════════════
#  4. METRICS
# ════════════════════════════════════════════════════════════════

@torch.no_grad()
def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold=0.3):
    probs = torch.sigmoid(logits).cpu()
    targets = targets.cpu()

    preds = (probs > threshold).float()

    tp = (preds * targets).sum()
    fp = (preds * (1 - targets)).sum()
    fn = ((1 - preds) * targets).sum()

    union = tp + fp + fn

    # CORRECTION MAJEURE ICI : Gérer les images qui ne contiennent pas de bidonvilles
    if union == 0:
        return {"iou": 1.0, "dice": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}

    iou = (tp / (union + 1e-7)).item()
    dice = (2 * tp / (2 * tp + fp + fn + 1e-7)).item()

    precision = (tp / (tp + fp + 1e-7)).item()
    recall = (tp / (tp + fn + 1e-7)).item()
    f1 = (2 * precision * recall / (precision + recall + 1e-7))

    return {"iou": iou, "dice": dice, "f1": f1, "precision": precision, "recall": recall}


# ════════════════════════════════════════════════════════════════
#  5. TRAINING & VALIDATION LOOPS
# ════════════════════════════════════════════════════════════════

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

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    n = len(loader)
    avg_loss = running_loss / n
    avg_metrics = {k: v / n for k, v in running_metrics.items()}
    return avg_loss, avg_metrics


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
    avg_loss = running_loss / n
    avg_metrics = {k: v / n for k, v in running_metrics.items()}
    return avg_loss, avg_metrics


# ════════════════════════════════════════════════════════════════
#  6. MAIN
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  Slum Segmentation — Training Pipeline")
    print("=" * 70)
    print(f"  Device      : {config.DEVICE}")
    print(f"  Architecture: {config.MODEL_ARCH}")
    print(f"  Batch size  : {config.BATCH_SIZE}")
    print(f"  LR          : {config.LEARNING_RATE}")
    print(f"  Epochs      : {config.NUM_EPOCHS}")
    print(f"  Patience    : {config.EARLY_STOPPING_PATIENCE}")
    print("=" * 70)

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    train_ds = SlumDataset(config.TRAIN_DIR, transform=get_train_transforms())
    val_ds = SlumDataset(config.VAL_DIR, transform=get_val_transforms())

    print(f"\n[DATA] Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=False,
    )

    model = build_model().to(config.DEVICE)

    criterion = CombinedLoss(
        bce_weight=config.BCE_WEIGHT,
        dice_weight=config.DICE_WEIGHT,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Cosine Annealing Learning Rate Scheduler for smoother convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    print("\n" + "─" * 70)
    print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>11}  "
          f"{'IoU':>7}  {'F1':>7}  {'PR':>7}  {'RC':>7}  {'LR':>10}  Status")
    print("─" * 70)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE,
        )
        val_loss, val_metrics = validate(
            model, val_loader, criterion, config.DEVICE,
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_metrics": val_metrics,
                "config": {
                    "model_arch": config.MODEL_ARCH,
                    "tile_size": config.TILE_SIZE,
                    "in_channels": config.IN_CHANNELS,
                    "num_classes": config.NUM_CLASSES,
                    "unet_encoder": config.UNET_ENCODER,
                },
            }, config.MODEL_SAVE_PATH)
            status = f"★ saved  ({elapsed:.1f}s)"
        else:
            patience_counter += 1
            status = f"  [{patience_counter}/{config.EARLY_STOPPING_PATIENCE}]  ({elapsed:.1f}s)"

        print(
            f"{epoch:>6d}  {train_loss:>11.4f}  {val_loss:>11.4f}  "
            f"{val_metrics['iou']:>7.4f}  {val_metrics['f1']:>7.4f}  "
            f"{val_metrics['precision']:>7.4f}  {val_metrics['recall']:>7.4f}  {current_lr:>10.2e}  {status}"
        )

        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print(f"\n[EARLY STOP] No improvement for {config.EARLY_STOPPING_PATIENCE} epochs. Stopping.")
            break

    print("─" * 70)
    print(f"\n[DONE] Best validation loss: {best_val_loss:.4f}")
    print(f"[DONE] Model saved to: {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()
# """
# train.py — Model definition, Dataset class, and full training loop.

# Usage:
#     python train.py

# Includes:
#     • SlumDataset       — PyTorch Dataset with albumentations augmentation
#     • CombinedLoss      — Dice Loss + BCE Loss
#     • build_model()     — SegFormer-B0 (default) or U-Net via smp
#     • compute_metrics() — IoU, Dice Score, F1 Score
#     • Training loop     — AdamW optimiser, early stopping, best-model checkpoint
# """

# import os
# import glob
# import time
# import numpy as np
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import segmentation_models_pytorch as smp
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# from tqdm import tqdm

# import config


# # ════════════════════════════════════════════════════════════════
# #  1. DATASET
# # ════════════════════════════════════════════════════════════════

# class SlumDataset(Dataset):
#     """
#     PyTorch Dataset for slum segmentation tiles.

#     Expects tiles saved as .npy files in:
#         {split_dir}/images/tile_XXXXX.npy   → uint8 (H, W, 3)
#         {split_dir}/masks/tile_XXXXX.npy    → uint8 (H, W), values {0, 1}
#     """

#     def __init__(self, split_dir: str, transform=None):
#         self.image_paths = sorted(glob.glob(os.path.join(split_dir, "images", "*.npy")))
#         self.mask_paths = sorted(glob.glob(os.path.join(split_dir, "masks", "*.npy")))

#         assert len(self.image_paths) == len(self.mask_paths), (
#             f"Mismatch: {len(self.image_paths)} images vs {len(self.mask_paths)} masks "
#             f"in {split_dir}"
#         )
#         assert len(self.image_paths) > 0, f"No tiles found in {split_dir}"

#         self.transform = transform

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         # Load .npy tiles
#         image = np.load(self.image_paths[idx])   # (H, W, 3) uint8
#         mask = np.load(self.mask_paths[idx])      # (H, W) uint8

#         # Apply augmentations (albumentations expects HWC uint8 images)
#         if self.transform:
#             augmented = self.transform(image=image, mask=mask)
#             image = augmented["image"]            # (3, H, W) float32, normalised
#             mask = augmented["mask"]              # (H, W) uint8/float
#         else:
#             # Manual fallback: normalise and convert to tensor
#             image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
#             mask = torch.from_numpy(mask)

#         # Ensure mask is float32 with shape (1, H, W) for BCEWithLogitsLoss
#         mask = mask.float().unsqueeze(0)

#         return image, mask


# # ────────────────────────────────────────────────────────────────
# #  Augmentation pipelines
# # ────────────────────────────────────────────────────────────────

# def get_train_transforms():
#     return A.Compose([
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.5),
#         A.RandomRotate90(p=0.5),
#         A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=(-45, 45), p=0.5), # Ajouté pour contrer l'overfitting
#         A.GaussianBlur(blur_limit=3, p=0.3), # Ajouté pour contrer l'overfitting
#         A.RandomBrightnessContrast(
#             brightness_limit=0.2,
#             contrast_limit=0.2,
#             p=0.5,
#         ),
#         A.Normalize(
#             mean=config.IMAGENET_MEAN,
#             std=config.IMAGENET_STD,
#         ),
#         ToTensorV2(),
#     ])


# def get_val_transforms():
#     return A.Compose([
#         A.Normalize(
#             mean=config.IMAGENET_MEAN,
#             std=config.IMAGENET_STD,
#         ),
#         ToTensorV2(),
#     ])


# # ════════════════════════════════════════════════════════════════
# #  2. MODEL
# # ════════════════════════════════════════════════════════════════

# def build_model() -> nn.Module:
#     """
#     Build the segmentation model based on config.MODEL_ARCH.

#     - "segformer": FPN decoder with Mix-Transformer-B0 encoder (SegFormer-B0).
#     - "unet":     U-Net with a configurable encoder backbone.
#     """
#     if config.MODEL_ARCH == "segformer":
#         print("[MODEL] SegFormer-B0  →  FPN(encoder=mit_b0)")
#         model = smp.FPN(
#             encoder_name="mit_b0",
#             encoder_weights="imagenet",
#             in_channels=config.IN_CHANNELS,
#             classes=config.NUM_CLASSES,
#         )
#     elif config.MODEL_ARCH == "unet":
#         print(f"[MODEL] U-Net  →  encoder={config.UNET_ENCODER}")
#         model = smp.Unet(
#             encoder_name=config.UNET_ENCODER,
#             encoder_weights="imagenet",
#             in_channels=config.IN_CHANNELS,
#             classes=config.NUM_CLASSES,
#         )
#     else:
#         raise ValueError(f"Unknown MODEL_ARCH: {config.MODEL_ARCH}. Use 'segformer' or 'unet'.")

#     total_params = sum(p.numel() for p in model.parameters())
#     trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"[MODEL] Parameters: {total_params:,} total  |  {trainable:,} trainable")

#     return model


# # ════════════════════════════════════════════════════════════════
# #  3. LOSS FUNCTION
# # ════════════════════════════════════════════════════════════════

# class CombinedLoss(nn.Module):
#     """
#     Weighted combination of Focal Loss and Dice Loss.
#     Focal Loss handles extreme class imbalance by focusing on hard-to-predict minority (slum) pixels.
#     """

#     def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0):
#         super().__init__()
#         self.focal_weight = bce_weight
#         self.dice_weight = dice_weight
#         # Use torchvision's native focal loss because smp.losses.FocalLoss 
#         # has a known bug with `output.type()` on the Apple MPS backend.
#         from torchvision.ops import sigmoid_focal_loss
#         self.focal_fn = sigmoid_focal_loss
#         self.dice = smp.losses.DiceLoss(mode=smp.losses.BINARY_MODE)

#     def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         focal_loss = self.focal_fn(logits, targets, alpha=0.25, gamma=2.0, reduction='mean')
#         dice_loss = self.dice(logits, targets)
#         return self.focal_weight * focal_loss + self.dice_weight * dice_loss


# # ════════════════════════════════════════════════════════════════
# #  4. METRICS
# # ════════════════════════════════════════════════════════════════

# @torch.no_grad()
# def compute_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold=0.3):
#     """
#     Compute IoU, Dice Score, and F1 Score for a batch of predictions.
#     Inputs:
#         logits  — raw model output (before sigmoid), any shape
#         targets — ground truth binary mask, same shape
#     Returns:
#         dict with keys: 'iou', 'dice', 'f1'
#     """
#     # Move to CPU to avoid potential MPS issues with reduction ops
#     probs = torch.sigmoid(logits).cpu()
#     targets = targets.cpu()

#     preds = (probs > threshold).float()

#     tp = (preds * targets).sum()
#     fp = (preds * (1 - targets)).sum()
#     fn = ((1 - preds) * targets).sum()

#     iou = (tp / (tp + fp + fn + 1e-7)).item()
#     dice = (2 * tp / (2 * tp + fp + fn + 1e-7)).item()

#     precision = (tp / (tp + fp + 1e-7)).item()
#     recall = (tp / (tp + fn + 1e-7)).item()
#     f1 = (2 * precision * recall / (precision + recall + 1e-7))

#     return {"iou": iou, "dice": dice, "f1": f1}


# # ════════════════════════════════════════════════════════════════
# #  5. TRAINING & VALIDATION LOOPS
# # ════════════════════════════════════════════════════════════════

# def train_one_epoch(model, loader, criterion, optimizer, device):
#     model.train()
#     running_loss = 0.0
#     running_metrics = {"iou": 0.0, "dice": 0.0, "f1": 0.0}

#     pbar = tqdm(loader, desc="  Train", leave=False, unit="batch")
#     for images, masks in pbar:
#         images = images.to(device)
#         masks = masks.to(device)

#         optimizer.zero_grad()
#         logits = model(images)
#         loss = criterion(logits, masks)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         batch_metrics = compute_metrics(logits, masks)
#         for k in running_metrics:
#             running_metrics[k] += batch_metrics[k]

#         pbar.set_postfix(loss=f"{loss.item():.4f}")

#     n = len(loader)
#     avg_loss = running_loss / n
#     avg_metrics = {k: v / n for k, v in running_metrics.items()}
#     return avg_loss, avg_metrics


# @torch.no_grad()
# def validate(model, loader, criterion, device):
#     model.eval()
#     running_loss = 0.0
#     running_metrics = {"iou": 0.0, "dice": 0.0, "f1": 0.0}

#     pbar = tqdm(loader, desc="  Val  ", leave=False, unit="batch")
#     for images, masks in pbar:
#         images = images.to(device)
#         masks = masks.to(device)

#         logits = model(images)
#         loss = criterion(logits, masks)

#         running_loss += loss.item()
#         batch_metrics = compute_metrics(logits, masks)
#         for k in running_metrics:
#             running_metrics[k] += batch_metrics[k]

#     n = len(loader)
#     avg_loss = running_loss / n
#     avg_metrics = {k: v / n for k, v in running_metrics.items()}
#     return avg_loss, avg_metrics


# # ════════════════════════════════════════════════════════════════
# #  6. MAIN
# # ════════════════════════════════════════════════════════════════

# def main():
#     print("=" * 70)
#     print("  Slum Segmentation — Training Pipeline")
#     print("=" * 70)
#     print(f"  Device      : {config.DEVICE}")
#     print(f"  Architecture: {config.MODEL_ARCH}")
#     print(f"  Batch size  : {config.BATCH_SIZE}")
#     print(f"  LR          : {config.LEARNING_RATE}")
#     print(f"  Epochs      : {config.NUM_EPOCHS}")
#     print(f"  Patience    : {config.EARLY_STOPPING_PATIENCE}")
#     print("=" * 70)

#     # ── Reproducibility ──────────────────────────────────────
#     torch.manual_seed(config.RANDOM_SEED)
#     np.random.seed(config.RANDOM_SEED)

#     # ── Datasets & DataLoaders ───────────────────────────────
#     train_ds = SlumDataset(config.TRAIN_DIR, transform=get_train_transforms())
#     val_ds = SlumDataset(config.VAL_DIR, transform=get_val_transforms())

#     print(f"\n[DATA] Train tiles: {len(train_ds)}  |  Val tiles: {len(val_ds)}")

#     train_loader = DataLoader(
#         train_ds,
#         batch_size=config.BATCH_SIZE,
#         shuffle=True,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=False,  # pin_memory not supported on MPS
#         drop_last=True,
#     )
#     val_loader = DataLoader(
#         val_ds,
#         batch_size=config.BATCH_SIZE,
#         shuffle=False,
#         num_workers=config.NUM_WORKERS,
#         pin_memory=False,
#     )

#     # ── Model ────────────────────────────────────────────────
#     model = build_model().to(config.DEVICE)

#     # ── Loss, Optimiser ──────────────────────────────────────
#     criterion = CombinedLoss(
#         bce_weight=config.BCE_WEIGHT,
#         dice_weight=config.DICE_WEIGHT,
#     )
#     optimizer = torch.optim.AdamW(
#         model.parameters(),
#         lr=config.LEARNING_RATE,
#         weight_decay=config.WEIGHT_DECAY,
#     )

#     # Learning rate scheduler (reduce on plateau)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode="min", factor=0.5, patience=7,
#     )

#     # ── Early stopping state ─────────────────────────────────
#     best_val_loss = float("inf")
#     patience_counter = 0
#     os.makedirs(config.OUTPUT_DIR, exist_ok=True)

#     # ── Training loop ────────────────────────────────────────
#     print("\n" + "─" * 70)
#     print(f"{'Epoch':>6}  {'Train Loss':>11}  {'Val Loss':>11}  "
#           f"{'IoU':>7}  {'Dice':>7}  {'F1':>7}  {'LR':>10}  Status")
#     print("─" * 70)

#     for epoch in range(1, config.NUM_EPOCHS + 1):
#         t0 = time.time()

#         train_loss, train_metrics = train_one_epoch(
#             model, train_loader, criterion, optimizer, config.DEVICE,
#         )
#         val_loss, val_metrics = validate(
#             model, val_loader, criterion, config.DEVICE,
#         )

#         scheduler.step(val_loss)
#         current_lr = optimizer.param_groups[0]["lr"]
#         elapsed = time.time() - t0

#         # Status flag
#         status = ""
#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#             torch.save({
#                 "epoch": epoch,
#                 "model_state_dict": model.state_dict(),
#                 "optimizer_state_dict": optimizer.state_dict(),
#                 "val_loss": val_loss,
#                 "val_metrics": val_metrics,
#                 "config": {
#                     "model_arch": config.MODEL_ARCH,
#                     "tile_size": config.TILE_SIZE,
#                     "in_channels": config.IN_CHANNELS,
#                     "num_classes": config.NUM_CLASSES,
#                     "unet_encoder": config.UNET_ENCODER,
#                 },
#             }, config.MODEL_SAVE_PATH)
#             status = f"★ saved  ({elapsed:.1f}s)"
#         else:
#             patience_counter += 1
#             status = f"  [{patience_counter}/{config.EARLY_STOPPING_PATIENCE}]  ({elapsed:.1f}s)"

#         print(
#             f"{epoch:>6d}  {train_loss:>11.4f}  {val_loss:>11.4f}  "
#             f"{val_metrics['iou']:>7.4f}  {val_metrics['dice']:>7.4f}  "
#             f"{val_metrics['f1']:>7.4f}  {current_lr:>10.2e}  {status}"
#         )

#         # Early stopping check
#         if patience_counter >= config.EARLY_STOPPING_PATIENCE:
#             print(f"\n[EARLY STOP] No improvement for {config.EARLY_STOPPING_PATIENCE} epochs. Stopping.")
#             break

#     print("─" * 70)
#     print(f"\n[DONE] Best validation loss: {best_val_loss:.4f}")
#     print(f"[DONE] Model saved to: {config.MODEL_SAVE_PATH}")

#     # ── Final evaluation on test set (if available) ──────────
#     test_dir = config.TEST_DIR
#     if os.path.isdir(os.path.join(test_dir, "images")) and \
#        len(glob.glob(os.path.join(test_dir, "images", "*.npy"))) > 0:
#         print("\n[TEST] Running evaluation on the test set …")
        
#         # Load best model
#         checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE, weights_only=False)
#         model.load_state_dict(checkpoint["model_state_dict"])

#         test_ds = SlumDataset(test_dir, transform=get_val_transforms())
#         test_loader = DataLoader(
#             test_ds,
#             batch_size=config.BATCH_SIZE,
#             shuffle=False,
#             num_workers=config.NUM_WORKERS,
#         )

#         test_loss, test_metrics = validate(model, test_loader, criterion, config.DEVICE)
#         print(f"[TEST] Loss: {test_loss:.4f}  |  IoU: {test_metrics['iou']:.4f}  |  "
#               f"Dice: {test_metrics['dice']:.4f}  |  F1: {test_metrics['f1']:.4f}")

#     print("\n" + "=" * 70)


# if __name__ == "__main__":
#     main()
