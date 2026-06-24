"""
Version 3: SegFormer-B2 with Lovasz Loss
This version achieved 95% Precision but 57% Recall.
"""
import torch
import segmentation_models_pytorch as smp

def get_v3_model():
    return smp.Unet(
        encoder_name="mit_b2",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
