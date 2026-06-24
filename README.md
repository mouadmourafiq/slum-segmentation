# Slum Settlement Semantic Segmentation
## Satellite Imagery Analysis — Casablanca

> **Deep learning pipeline for detecting slum settlements in satellite imagery using SegFormer-B0, optimised for Apple M1 Pro with MPS acceleration.**

---

## Project Overview

This project performs **binary semantic segmentation** to identify slum settlements from high-resolution satellite imagery of Casablanca. The pipeline tiles large GeoTIFFs into training patches, trains a SegFormer-B0 model, and reconstructs full-resolution prediction maps with preserved geospatial metadata.

### Architecture

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Encoder** | Mix Transformer B0 (mit_b0) | SegFormer-B0 encoder — lightweight, ~3.7M params |
| **Decoder** | FPN (Feature Pyramid Network) | Efficient multi-scale feature aggregation |
| **Loss** | Dice + BCE (weighted) | Handles class imbalance (slum pixels are minority) |
| **Optimiser** | AdamW | Better weight decay regularisation |

---

## Requirements

- **Python** 3.11+
- **Hardware**: Apple M1/M2/M3 Pro (16 GB+ Unified Memory) — or any CUDA GPU
- **PyTorch** ≥ 2.1 (includes MPS backend for Apple Silicon)

## Quick Start

### 1. Clone & Install

```bash
cd slum_segmentation
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Your Data

Place your GeoTIFF files in the `data/` directory:

```
slum_segmentation/
└── data/
    ├── casablancasansmask.tif    ← RGB satellite image
    └── mask_global.tif           ← Ground truth binary mask
```

> **Tip:** You can also create symlinks:
> ```bash
> mkdir -p data
> ln -s /path/to/casablancasansmask.tif data/
> ln -s /path/to/mask_global.tif data/
> ```

### 3. Tile & Split the Data

```bash
python prepare_data.py
```

This will:
- Read both large GeoTIFFs using efficient windowed reads
- Normalise the mask to binary {0, 1} (with optional inversion)
- Generate 512×512 pixel tiles
- Filter out empty/black tiles
- Split into **Train (80%)** / **Val (10%)** / **Test (10%)**
- Save tiles as `.npy` files in `data/tiles/{train,val,test}/`

### 4. Train the Model

```bash
python train.py
```

Training will:
- Use **SegFormer-B0** by default (configurable to U-Net)
- Apply data augmentation (flips, rotations, brightness/contrast)
- Print **IoU**, **Dice Score**, and **F1** at each epoch
- Save `best_model.pth` based on best validation loss
- Stop early if no improvement for 15 epochs

### 5. Run Inference

```bash
python inference.py
```

Or with custom arguments:

```bash
python inference.py \
    --input data/casablancasansmask.tif \
    --model output/best_model.pth \
    --output output/predicted_slums.tif \
    --tile-size 512 \
    --overlap 64 \
    --threshold 0.5
```

This produces:
- `predicted_slums.tif` — Binary prediction mask (GeoTIFF with CRS/Transform)
- `predicted_slums_probabilities.tif` — Probability map (float32)

---

## Project Structure

```
slum_segmentation/
├── config.py            # All paths, hyperparameters, device selection
├── prepare_data.py      # Tiling, filtering, train/val/test splitting
├── train.py             # Dataset, model, loss, training loop
├── inference.py         # Full-image prediction & GeoTIFF export
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── data/                # Input GeoTIFFs (user-provided)
│   ├── casablancasansmask.tif
│   ├── mask_global.tif
│   └── tiles/           # Generated tiles (auto-created)
│       ├── train/
│       │   ├── images/
│       │   └── masks/
│       ├── val/
│       └── test/
└── output/              # Training outputs (auto-created)
    ├── best_model.pth
    ├── predicted_slums.tif
    └── predicted_slums_probabilities.tif
```

---

## Configuration

All tuneable parameters live in [`config.py`](config.py). Key settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL_ARCH` | `"segformer"` | `"segformer"` (SegFormer-B0) or `"unet"` |
| `UNET_ENCODER` | `"efficientnet-b0"` | Encoder backbone when using U-Net |
| `TILE_SIZE` | `512` | Patch size for tiling |
| `BATCH_SIZE` | `4` | Training batch size (M1 Pro optimised) |
| `LEARNING_RATE` | `1e-4` | Initial learning rate for AdamW |
| `NUM_EPOCHS` | `100` | Maximum training epochs |
| `EARLY_STOPPING_PATIENCE` | `15` | Epochs without improvement before stopping |
| `INVERT_MASK` | `True` | Set `True` if mask has slum=0, background=255 |
| `INFERENCE_OVERLAP` | `64` | Overlap pixels for seamless prediction stitching |
| `CONFIDENCE_THRESHOLD` | `0.5` | Probability threshold for binary output |

---

## Mask Handling

The pipeline handles multiple mask formats:

| Original Mask | `INVERT_MASK` | Result |
|---------------|---------------|--------|
| 0 = slum, 255 = background | `True` | ✅ Correct: 0=bg, 1=slum |
| 0 = background, 255 = slum | `False` | ✅ Correct: 0=bg, 1=slum |
| Already binary {0, 1} with 1=slum | `False` | ✅ Correct |

Based on the provided mask (black patches on white background), `INVERT_MASK = True` is the correct default.

---

## Device Selection

The pipeline automatically selects the best available accelerator:

```
MPS (Apple Metal)  →  CUDA (NVIDIA)  →  CPU
```

On your M1 Pro MacBook, it will use **MPS** automatically. No manual configuration needed.

---

## Metrics

The training loop tracks three key metrics per epoch:

- **IoU** (Intersection over Union) — overlap between predicted and ground truth
- **Dice Score** — F1-equivalent for segmentation masks
- **F1 Score** — harmonic mean of precision and recall

---

## Troubleshooting

### MPS Issues
If you encounter MPS-related errors, you can force CPU mode by editing `config.py`:
```python
DEVICE = torch.device("cpu")
```

### Memory Issues
If you run out of memory, reduce the batch size in `config.py`:
```python
BATCH_SIZE = 2  # or even 1
```

### Empty Tiles
If too many tiles are being discarded, lower the threshold in `config.py`:
```python
MIN_VALID_RATIO = 0.005  # keep tiles with ≥ 0.5% non-black pixels
```

---

## License

This project is for research and educational purposes.
