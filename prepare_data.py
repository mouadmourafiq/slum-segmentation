"""
prepare_data.py — Tile large GeoTIFFs and split into Train / Val / Test sets.
Supports Geographic Holdout (Leave-One-City-Out) to prevent Data Leakage.
"""

import os
import json
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config


# ────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────

def create_directory_tree():
    """Create the output directory structure for tiles."""
    for split in ("train", "val", "test"):
        for sub in ("images", "masks"):
            path = os.path.join(config.TILES_DIR, split, sub)
            os.makedirs(path, exist_ok=True)
    print("[INFO] Directory tree created under:", config.TILES_DIR)


def verify_mask_binary(mask_path: str) -> dict:
    with rasterio.open(mask_path) as src:
        sample_window = Window(0, 0, min(1024, src.width), min(1024, src.height))
        sample = src.read(1, window=sample_window)
        info = {
            "dtype": str(src.dtypes[0]),
            "width": src.width,
            "height": src.height,
            "bands": src.count,
            "crs": str(src.crs),
            "sample_min": int(sample.min()),
            "sample_max": int(sample.max()),
            "sample_unique": sorted(set(np.unique(sample).tolist()[:20])),
        }
    return info


def normalise_mask_patch(patch: np.ndarray) -> np.ndarray:
    patch = patch.astype(np.float32)
    if patch.max() > 1.0:
        patch = patch / 255.0
    patch = (patch > 0.5).astype(np.uint8)
    if config.INVERT_MASK:
        patch = 1 - patch
    return patch


def is_tile_empty(image_patch: np.ndarray) -> bool:
    total_pixels = image_patch.size
    nonzero_pixels = np.count_nonzero(image_patch)
    ratio = nonzero_pixels / total_pixels
    return ratio < config.MIN_VALID_RATIO


# ────────────────────────────────────────────────────────────────
#  Main tiling routine
# ────────────────────────────────────────────────────────────────

def extract_valid_tiles(city_name: str, img_path: str, msk_path: str, global_tile_idx: int):
    """Generate tiles in memory for a single city, limited to MAX_TILES_PER_CITY."""
    tile_size = config.TILE_SIZE
    stride = config.STRIDE

    src_img = rasterio.open(img_path)
    src_msk = rasterio.open(msk_path)

    img_h, img_w = src_img.height, src_img.width
    msk_h, msk_w = src_msk.height, src_msk.width
    
    height = min(img_h, msk_h)
    width = min(img_w, msk_w)

    all_coords = [(x, y) for y in range(0, height, stride) for x in range(0, width, stride)]
    
    print(f"\n[INFO] [{city_name}] Scanning {len(all_coords)} candidate patches ...")
    slum_coords = []
    bg_coords = []
    
    for x, y in tqdm(all_coords, desc=f"Scanning mask {city_name}", leave=False):
        read_w = int(min(tile_size, width - x))
        read_h = int(min(tile_size, height - y))
        if read_w < tile_size * 0.25 or read_h < tile_size * 0.25:
            continue
            
        window = Window(col_off=x, row_off=y, width=read_w, height=read_h)
        msk_data = src_msk.read(1, window=window)
        
        if msk_data.max() > 1.0:
            msk_data = msk_data / 255.0
        msk_data = (msk_data > 0.5).astype(np.uint8)
        if config.INVERT_MASK:
            msk_data = 1 - msk_data
            
        if np.any(msk_data > 0):
            slum_coords.append((x, y))
        else:
            bg_coords.append((x, y))
            
    print(f"[INFO] [{city_name}] Found {len(slum_coords)} patches containing slums and {len(bg_coords)} background patches.")
    
    np.random.seed(config.RANDOM_SEED)
    np.random.shuffle(slum_coords)
    np.random.shuffle(bg_coords)
    
    # HARD NEGATIVE MINING: Force a 50/50 split if possible to prevent paranoia
    half_max = config.MAX_TILES_PER_CITY // 2
    take_slum = min(len(slum_coords), half_max)
    take_bg = min(len(bg_coords), config.MAX_TILES_PER_CITY - take_slum)
    
    selected_coords = slum_coords[:take_slum] + bg_coords[:take_bg]
    np.random.shuffle(selected_coords)

    valid_tiles = []
    skipped = 0
    pbar = tqdm(total=min(config.MAX_TILES_PER_CITY, len(selected_coords)), desc=f"Extracting {city_name}", unit="tile")

    for x, y in selected_coords:
        if len(valid_tiles) >= config.MAX_TILES_PER_CITY:
            break

        read_w = int(min(tile_size, width - x))
        read_h = int(min(tile_size, height - y))

        if read_w < tile_size * 0.25 or read_h < tile_size * 0.25:
            skipped += 1
            continue

        window = Window(col_off=x, row_off=y, width=read_w, height=read_h)
        img_data = src_img.read(window=window)[:3, :, :]
        msk_data = src_msk.read(1, window=window)

        if read_h < tile_size or read_w < tile_size:
            padded_img = np.zeros((3, tile_size, tile_size), dtype=img_data.dtype)
            padded_img[:, :read_h, :read_w] = img_data
            img_data = padded_img

            padded_msk = np.zeros((tile_size, tile_size), dtype=msk_data.dtype)
            padded_msk[:read_h, :read_w] = msk_data
            msk_data = padded_msk

        if is_tile_empty(img_data):
            skipped += 1
            continue

        msk_data = normalise_mask_patch(msk_data)
        img_hwc = np.transpose(img_data, (1, 2, 0))

        name = f"tile_{city_name.replace(' ', '_')}_{global_tile_idx:05d}"
        
        valid_tiles.append({
            "name": name,
            "img": img_hwc,
            "msk": msk_data
        })
        
        global_tile_idx += 1
        pbar.update(1)

    pbar.close()
    src_img.close()
    src_msk.close()

    print(f"[RESULT] [{city_name}] Extracted {len(valid_tiles)} tiles.")
    return valid_tiles, global_tile_idx


def process_and_save_data():
    create_directory_tree()

    global_tile_idx = 0
    stats = {"train": 0, "val": 0, "test": 0}
    
    for city, (img_name, msk_name) in config.MULTI_CITY_FILES.items():
        img_path = os.path.join(config.MULTI_CITY_DATA_DIR, city, img_name)
        msk_path = os.path.join(config.MULTI_CITY_DATA_DIR, city, msk_name)
        
        tiles, global_tile_idx = extract_valid_tiles(city, img_path, msk_path, global_tile_idx)
        
        # Geographic split logic
        if city in config.TRAIN_CITIES:
            # Split this city's tiles into train and val
            train_tiles, val_tiles = train_test_split(
                tiles, test_size=config.VAL_RATIO, random_state=config.RANDOM_SEED
            )
            for split, split_tiles in [("train", train_tiles), ("val", val_tiles)]:
                for t in split_tiles:
                    np.save(os.path.join(config.TILES_DIR, split, "images", f"{t['name']}.npy"), t["img"])
                    np.save(os.path.join(config.TILES_DIR, split, "masks", f"{t['name']}.npy"), t["msk"])
                stats[split] += len(split_tiles)
                
        elif city in config.TEST_CITIES:
            # 100% goes to test
            for t in tiles:
                np.save(os.path.join(config.TILES_DIR, "test", "images", f"{t['name']}.npy"), t["img"])
                np.save(os.path.join(config.TILES_DIR, "test", "masks", f"{t['name']}.npy"), t["msk"])
            stats["test"] += len(tiles)
            
        else:
            print(f"[WARNING] City {city} is neither in TRAIN_CITIES nor TEST_CITIES. Skipping.")

    print(f"\n[SPLIT] Train: {stats['train']}  |  Val: {stats['val']}  |  Test: {stats['test']}")
    
    info_path = os.path.join(config.TILES_DIR, "split_info.json")
    with open(info_path, "w") as f:
        json.dump(stats, f, indent=2)


# ────────────────────────────────────────────────────────────────
#  Entry point
# ────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Slum Segmentation — EXPERT Pipeline: Data Preparation")
    print("  (Geographical Holdout / Zero Data Leakage)")
    print("=" * 64)

    for city, (img_name, msk_name) in config.MULTI_CITY_FILES.items():
        img_path = os.path.join(config.MULTI_CITY_DATA_DIR, city, img_name)
        msk_path = os.path.join(config.MULTI_CITY_DATA_DIR, city, msk_name)
        if not os.path.isfile(img_path):
            raise FileNotFoundError(f"Image not found at: {img_path}")
        if not os.path.isfile(msk_path):
            raise FileNotFoundError(f"Mask not found at: {msk_path}")
        
        split = "TRAIN/VAL" if city in config.TRAIN_CITIES else "TEST (Holdout)" if city in config.TEST_CITIES else "IGNORED"
        print(f"[OK] {city} files found. Role: {split}")

    process_and_save_data()

    print("\n[DONE] Data preparation complete. Ready for training.")
    print("=" * 64)


if __name__ == "__main__":
    main()
