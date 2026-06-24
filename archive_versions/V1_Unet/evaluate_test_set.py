"""
evaluate_test_set.py — Rigorous evaluation on the held-out test set (Zero Data Leakage).
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from train import SlumDataset, get_val_transforms, build_model, compute_metrics


@torch.no_grad()
def evaluate_test_set():
    print("=" * 70)
    print("  Slum Segmentation — EXPERT Test Set Evaluation")
    print("  (Zero-Shot / Geographic Holdout)")
    print("=" * 70)

    test_dir = config.TEST_DIR
    if not os.path.isdir(os.path.join(test_dir, "images")):
        print("[❌ ERROR] Test directory not found. Did you run prepare_data.py?")
        return

    num_test_files = len(glob.glob(os.path.join(test_dir, "images", "*.npy")))
    if num_test_files == 0:
        print("[❌ ERROR] No tiles found in test directory.")
        return

    print(f"[INFO] Found {num_test_files} test tiles (from {config.TEST_CITIES}).")

    if not os.path.isfile(config.MODEL_SAVE_PATH):
        print(f"[❌ ERROR] Model checkpoint not found at: {config.MODEL_SAVE_PATH}")
        return

    device = config.DEVICE
    print(f"[INFO] Using device: {device}")

    # Load dataset
    test_ds = SlumDataset(test_dir, transform=get_val_transforms())
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
    )

    # Load model architecture and weights
    print("[INFO] Loading trained model weights...")
    model = build_model()
    checkpoint = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    running_metrics = {"iou": 0.0, "dice": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    # Redefine metric computation to include precision and recall
    def compute_full_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold=0.3):
        probs = torch.sigmoid(logits).cpu()
        targets = targets.cpu()
        preds = (probs > threshold).float()

        tp = (preds * targets).sum()
        fp = (preds * (1 - targets)).sum()
        fn = ((1 - preds) * targets).sum()

        union = tp + fp + fn
        if union == 0:
            return {"iou": 1.0, "dice": 1.0, "f1": 1.0, "precision": 1.0, "recall": 1.0}

        iou = (tp / (union + 1e-7)).item()
        dice = (2 * tp / (2 * tp + fp + fn + 1e-7)).item()
        precision = (tp / (tp + fp + 1e-7)).item()
        recall = (tp / (tp + fn + 1e-7)).item()
        f1 = (2 * precision * recall / (precision + recall + 1e-7)) if (precision + recall) > 0 else 0.0

        return {"iou": iou, "dice": dice, "f1": f1, "precision": precision, "recall": recall}

    print("\n[INFO] Evaluating on the isolated test set...")
    pbar = tqdm(test_loader, desc="Testing", unit="batch")
    
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        batch_metrics = compute_full_metrics(logits, masks)
        
        for k in running_metrics:
            running_metrics[k] += batch_metrics[k]

    n = len(test_loader)
    avg_metrics = {k: v / n for k, v in running_metrics.items()}

    print("\n" + "─" * 70)
    print("  🏆 FINAL HOLDOUT RESULTS (True Generalisation)")
    print("─" * 70)
    print(f"  F1-Score (Dice) : {avg_metrics['f1']:.4f}")
    print(f"  IoU             : {avg_metrics['iou']:.4f}")
    print(f"  Precision       : {avg_metrics['precision']:.4f}")
    print(f"  Recall          : {avg_metrics['recall']:.4f}")
    print("─" * 70)
    print("[DONE] Evaluation complete.")

if __name__ == "__main__":
    evaluate_test_set()
