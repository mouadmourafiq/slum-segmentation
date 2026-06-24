import rasterio
from rasterio.windows import Window
import os

def crop_center_40(input_path, output_path):
    print(f"Opening {input_path}...")
    with rasterio.open(input_path) as src:
        width = src.width
        height = src.height
        
        # We want the center 40%.
        # So we skip the first 30% and take the next 40%. (30% + 40% + 30% = 100%)
        col_off = int(width * 0.30)
        row_off = int(height * 0.30)
        new_width = int(width * 0.40)
        new_height = int(height * 0.40)
        
        window = Window(col_off, row_off, new_width, new_height)
        
        print(f"Original size: {width}x{height}")
        print(f"Cropped size: {new_width}x{new_height}")
        
        kwargs = src.meta.copy()
        kwargs.update({
            'height': new_height,
            'width': new_width,
            'transform': rasterio.windows.transform(window, src.transform)
        })
        
        with rasterio.open(output_path, 'w', **kwargs) as dst:
            dst.write(src.read(window=window))
            
    print(f"Saved cropped image to {output_path}")

if __name__ == "__main__":
    base_dir = "/Users/oussamalouat/Documents/slum_segmentation/data"
    
    # Crop the satellite image
    img_in = os.path.join(base_dir, "image_cropped.tif")
    img_out = os.path.join(base_dir, "image_center_40.tif")
    
    if os.path.exists(img_in):
        crop_center_40(img_in, img_out)
    else:
        print(f"Error: {img_in} not found.")
