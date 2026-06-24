"""
vectorize.py — GIS Vectorization Script
Converts the output raster mask (.tif) into geographical polygons (.geojson)
for use in QGIS, ArcGIS, or web mapping tools.
"""

import os
import argparse
import numpy as np
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
import geopandas as gpd
import json

import config


def vectorize_mask(input_tiff, output_geojson=None):
    print("=" * 64)
    print("  Slum Segmentation — GIS VECTORIZATION")
    print("=" * 64)

    if not os.path.exists(input_tiff):
        raise FileNotFoundError(f"Predicted mask not found at {input_tiff}. Please run inference first.")

    print(f"[INFO] Reading raster file: {input_tiff}")
    with rasterio.open(input_tiff) as src:
        image = src.read(1)  # Read the first (and only) band
        transform = src.transform
        crs = src.crs

    # The inference script saves probabilities as 0 and 1 uint8.
    # We binarize it strictly: > 0 is slum
    binary_mask = (image > 0).astype("uint8")

    print("[INFO] Extracting polygons using rasterio.features.shapes...")
    # `shapes` returns a generator of (polygon_dict, value)
    polygons = []
    for geom, val in shapes(binary_mask, mask=binary_mask, transform=transform):
        if val == 1:  # Keep only the slum regions (value=1)
            poly = shape(geom)
            # Optional: Smooth the polygon slightly to make it look nicer in GIS
            poly = poly.simplify(0.5, preserve_topology=True)
            polygons.append(poly)

    if len(polygons) == 0:
        print("[WARNING] No slum areas found in the mask. GeoJSON will be empty.")
    else:
        print(f"[INFO] Found {len(polygons)} distinct slum regions.")

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame({"geometry": polygons, "label": ["slum"] * len(polygons)})
    
    # Assign the Coordinate Reference System (CRS)
    if crs is not None:
        gdf.set_crs(crs, inplace=True)
        print(f"[INFO] Preserved native CRS: {crs}")
        # Reproject to standard EPSG:4326 (Lat/Lon) for Web Mapping (Leaflet/Mapbox)
        print("[INFO] Reprojecting to EPSG:4326 for Web compatibility...")
        gdf = gdf.to_crs(epsg=4326)
    else:
        print("[WARNING] No native CRS found in the TIFF. Polygons are in pixel coordinates.")

    print(f"[INFO] Exporting to GeoJSON: {output_geojson}...")
    
    if output_geojson:
        # Handle existing files safely
        if os.path.exists(output_geojson):
            os.remove(output_geojson)
            
        gdf.to_file(output_geojson, driver="GeoJSON")
        print("\n[DONE] Vectorization complete! The file is ready for QGIS/Google Earth.")
        print("=" * 64)
        
    # Return as JSON dict
    return json.loads(gdf.to_json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Raster mask to GeoJSON polygons")
    parser.add_argument("--input", type=str, default=config.PREDICTION_OUTPUT_PATH, help="Path to input .tif mask")
    
    # We create a generic output path based on the input name
    default_output = config.PREDICTION_OUTPUT_PATH.replace(".tif", ".geojson")
    parser.add_argument("--output", type=str, default=default_output, help="Path to output .geojson file")
    
    args = parser.parse_args()
    vectorize_mask(args.input, args.output)
