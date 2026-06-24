"""
inference.py — Run trained model on a full-size GeoTIFF using Smooth Sliding Window.
"""

import os
import argparse
import numpy as np
import torch
import rasterio
from rasterio.windows import Window
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
from tqdm import tqdm

import config


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved_config = checkpoint.get("config", {})

    model_arch = saved_config.get("model_arch", config.MODEL_ARCH)
    in_channels = saved_config.get("in_channels", config.IN_CHANNELS)
    num_classes = saved_config.get("num_classes", config.NUM_CLASSES)

    if model_arch == "segformer":
        model = smp.FPN(
            encoder_name="mit_b0",
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
    elif model_arch == "unet":
        encoder = saved_config.get("unet_encoder", config.UNET_ENCODER)
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights=None,
            in_channels=in_channels,
            classes=num_classes,
        )
    else:
        raise ValueError(f"Unknown model architecture: {model_arch}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def get_inference_transform():
    return A.Compose([
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def create_gaussian_weight(tile_size: int, sigma: float = 0.25) -> np.ndarray:
    """Create a 2D Gaussian weight matrix for blending overlapping tiles."""
    x = np.linspace(-1, 1, tile_size)
    y = np.linspace(-1, 1, tile_size)
    xx, yy = np.meshgrid(x, y)
    weight = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return weight.astype(np.float32)


@torch.no_grad()
def run_inference_smooth(
    model: torch.nn.Module,
    input_path: str,
    output_path: str,
    device: torch.device,
    tile_size: int = 512,
    overlap: float = 0.5,
    threshold: float = 0.5,
):
    """
    Run patch-by-patch inference using a sliding window with overlap and Gaussian blending.
    This entirely eliminates grid edge artifacts.
    """
    transform = get_inference_transform()
    stride = int(tile_size * (1 - overlap))
    if stride <= 0:
        stride = tile_size // 2

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    weight_kernel = create_gaussian_weight(tile_size)

    with rasterio.open(input_path) as src:
        img_height = src.height
        img_width = src.width
        img_crs = src.crs
        img_transform = src.transform
        num_bands = src.count

        print(f"[INPUT]  {input_path}")
        print(f"         Size: {img_width} × {img_height} px  |  Bands: {num_bands}  |  CRS: {img_crs}")

        # Allocate full-size memory maps for probability and weight accumulation
        # Using memory-mapped files to avoid RAM explosions on huge images
        prob_map_path = output_path.replace(".tif", "_prob_map.dat")
        weight_map_path = output_path.replace(".tif", "_weight_map.dat")
        
        prob_map = np.memmap(prob_map_path, dtype='float32', mode='w+', shape=(img_height, img_width))
        weight_map = np.memmap(weight_map_path, dtype='float32', mode='w+', shape=(img_height, img_width))
        
        y_starts = list(range(0, max(1, img_height - tile_size + 1), stride))
        if y_starts[-1] + tile_size < img_height:
            y_starts.append(img_height - tile_size)
            
        x_starts = list(range(0, max(1, img_width - tile_size + 1), stride))
        if x_starts[-1] + tile_size < img_width:
            x_starts.append(img_width - tile_size)

        total_tiles = len(y_starts) * len(x_starts)
        print(f"[INFER]  Tile size: {tile_size}  |  Stride: {stride}  |  Overlap: {overlap*100}%")
        print(f"[INFER]  Grid: {len(x_starts)} × {len(y_starts)} = {total_tiles} tiles\n")

        pbar = tqdm(total=total_tiles, desc="Inference (Sliding Window)", unit="tile")

        for y in y_starts:
            for x in x_starts:
                window = Window(col_off=x, row_off=y, width=tile_size, height=tile_size)
                tile_data = src.read(window=window)

                # Keep only RGB and handle edges
                tile_rgb = tile_data[:3, :, :]
                _, read_h, read_w = tile_rgb.shape

                if read_h < tile_size or read_w < tile_size:
                    padded = np.zeros((3, tile_size, tile_size), dtype=tile_rgb.dtype)
                    padded[:, :read_h, :read_w] = tile_rgb
                    tile_rgb = padded

                tile_hwc = np.transpose(tile_rgb, (1, 2, 0))
                augmented = transform(image=tile_hwc)
                input_tensor = augmented["image"].unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.sigmoid(logits).squeeze().cpu().numpy()

                # Accumulate probabilities and weights
                prob_map[y:y+read_h, x:x+read_w] += (probs * weight_kernel)[:read_h, :read_w]
                weight_map[y:y+read_h, x:x+read_w] += weight_kernel[:read_h, :read_w]

                pbar.update(1)

        pbar.close()

        print("\n[INFO] Thresholding and exporting final map...")
        # To avoid division by zero
        weight_map[weight_map == 0] = 1.0
        prob_map[:] = prob_map / weight_map
        
        binary_mask = (prob_map > threshold).astype(np.uint8)
        slum_pixels = binary_mask.sum()
        total_pixels = img_width * img_height

        out_meta = {
            "driver": "GTiff",
            "dtype": "uint8",
            "width": img_width,
            "height": img_height,
            "count": 1,
            "crs": img_crs,
            "transform": img_transform,
            "compress": "lzw",
            "nodata": 255,
        }

        with rasterio.open(output_path, "w", **out_meta) as dst:
            dst.set_band_description(1, "slum_prediction")
            dst.write(binary_mask, 1)

    # Cleanup temp memmap files
    del prob_map
    del weight_map
    if os.path.exists(prob_map_path): os.remove(prob_map_path)
    if os.path.exists(weight_map_path): os.remove(weight_map_path)

    slum_pct = 100.0 * slum_pixels / total_pixels
    print(f"\n[STATS] Slum pixels: {slum_pixels:,} / {total_pixels:,} ({slum_pct:.2f}%)")
    print(f"[OUTPUT] Saved smooth prediction to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Smooth slum segmentation inference.")
    parser.add_argument("--input", type=str, default=config.INPUT_IMAGE_PATH)
    parser.add_argument("--model", type=str, default=config.MODEL_SAVE_PATH)
    parser.add_argument("--output", type=str, default=config.PREDICTION_OUTPUT_PATH)
    parser.add_argument("--tile-size", type=int, default=config.INFERENCE_TILE_SIZE)
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap ratio (e.g. 0.5 for 50%)")
    parser.add_argument("--threshold", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--adapted", action="store_true", help="Use the domain-adapted model instead of the universal one")
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 64)
    if args.adapted:
        print("  Slum Segmentation — INFERENCE (GRAM ADAPTED MODEL)")
        model_path = config.MODEL_SAVE_PATH.replace(".pth", "_adapted.pth")
    else:
        print("  Slum Segmentation — INFERENCE (UNIVERSAL MODEL)")
        model_path = args.model
    print("=" * 64)
    
    device = config.DEVICE
    print(f"  Device    : {device}")
    print(f"  Model     : {model_path}")
    print(f"  Overlap   : {args.overlap*100}% (Smooth Blending)")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.isfile(args.input):
        raise FileNotFoundError(f"Input image not found: {args.input}")

    model = load_model(model_path, device)
    
    run_inference_smooth(
        model=model,
        input_path=args.input,
        output_path=args.output,
        device=device,
        tile_size=args.tile_size,
        overlap=args.overlap,
        threshold=args.threshold,
    )
    print("=" * 70)

if __name__ == "__main__":
    main()
