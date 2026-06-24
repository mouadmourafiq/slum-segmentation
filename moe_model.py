import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

class RegionAwareMoE(nn.Module):
    """
    Region-Aware Mixture-of-Experts (MoE) Architecture inspired by GRAM.
    """
    def __init__(self, num_experts=3, model_name="mit_b0"):
        super().__init__()
        self.num_experts = num_experts
        
        self.experts = nn.ModuleList([
            smp.FPN(
                encoder_name=model_name,
                encoder_weights="imagenet",
                in_channels=3,
                classes=1
            ) for _ in range(num_experts)
        ])
        
        self.router = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, num_experts, kernel_size=3, padding=1)
        )
        
    def forward(self, pixel_values, labels=None):
        expert_outputs = []
        for expert in self.experts:
            # smp returns logits of shape (B, 1, H, W)
            out = expert(pixel_values)
            expert_outputs.append(out)
            
        stacked_expert_logits = torch.stack(expert_outputs, dim=1) # (B, N, 1, H, W)
        
        router_logits = self.router(pixel_values) # (B, N, H/4, W/4)
        
        # Upsample router weights to full image size (H, W) to match expert logits
        router_logits = F.interpolate(router_logits, size=pixel_values.shape[-2:], mode='bilinear', align_corners=False) # (B, N, H, W)
        
        router_weights = F.softmax(router_logits, dim=1)
        router_weights = router_weights.unsqueeze(2) # (B, N, 1, H, W)
        
        mixed_logits = torch.sum(stacked_expert_logits * router_weights, dim=1) # (B, 1, H, W)
        
        return mixed_logits
