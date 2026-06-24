import rasterio
import numpy as np
import os

data_dir = "/Users/oussamalouat/Documents/data set"
cities = ["Colombia", "Nigeria Makoko", "sudan ElGeneina"]
files = {
    "Colombia": ("Medellin_40cm.tif", "Medellin_ground_truth.tif"),
    "Nigeria Makoko": ("Makoko_50cm_small.tif", "Makoko_50cm_small_ground_truth.tif"),
    "sudan ElGeneina": ("ElGeneina_40cm.tif", "ElGeneina_40cm_ground_truth.tif"),
}

for city in cities:
    print(f"\n--- {city} ---")
    img_name, mask_name = files[city]
    img_path = os.path.join(data_dir, city, img_name)
    mask_path = os.path.join(data_dir, city, mask_name)
    
    with rasterio.open(img_path) as src:
        print(f"Image: {img_name}")
        print(f"  Shape: {src.shape}, Channels: {src.count}")
        print(f"  Dtype: {src.dtypes[0]}")
    
    with rasterio.open(mask_path) as src:
        print(f"Mask: {mask_name}")
        print(f"  Shape: {src.shape}, Channels: {src.count}")
        mask_data = src.read(1)
        unique_vals = np.unique(mask_data)
        print(f"  Unique Values: {unique_vals}")
