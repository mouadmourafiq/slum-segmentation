"""
Crop les deux fichiers GeoTIFF (image satellite + masque) autour de la zone
contenant les bidonvilles. Cela réduit drastiquement le déséquilibre des classes
et accélère tout le pipeline (tiling, entraînement, inférence).
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from tqdm import tqdm

def find_slum_bbox(mask_path, block_size=1024):
    """Scanne le masque par blocs pour trouver le rectangle englobant des bidonvilles."""
    print(f"[INFO] Scan du masque pour localiser les bidonvilles...")
    
    min_y, min_x = float('inf'), float('inf')
    max_y, max_x = 0, 0
    slum_found = False
    
    with rasterio.open(mask_path) as src:
        height = src.height
        width = src.width
        
        total_blocks = ((height + block_size - 1) // block_size) * ((width + block_size - 1) // block_size)
        pbar = tqdm(total=total_blocks, desc="Scanning", unit="block")
        
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                read_h = min(block_size, height - y)
                read_w = min(block_size, width - x)
                
                window = Window(col_off=x, row_off=y, width=read_w, height=read_h)
                block = src.read(1, window=window)
                
                # Normaliser
                if block.max() > 1.0:
                    block = block / 255.0
                binary = (block > 0.5).astype(np.uint8)
                
                if np.any(binary > 0):
                    slum_found = True
                    rows, cols = np.where(binary > 0)
                    min_y = min(min_y, y + rows.min())
                    max_y = max(max_y, y + rows.max())
                    min_x = min(min_x, x + cols.min())
                    max_x = max(max_x, x + cols.max())
                
                pbar.update(1)
        pbar.close()
    
    if not slum_found:
        print("[❌ ERREUR] Aucun pixel de bidonville trouvé dans le masque !")
        return None
    
    return min_x, min_y, max_x, max_y


def crop_geotiff(input_path, output_path, window):
    """Recadre un GeoTIFF selon une fenêtre donnée."""
    with rasterio.open(input_path) as src:
        # Lire les données dans la fenêtre
        data = src.read(window=window)
        
        # Calculer la nouvelle transformation géographique
        new_transform = rasterio.windows.transform(window, src.transform)
        
        # Mettre à jour les métadonnées
        meta = src.meta.copy()
        meta.update({
            'width': int(window.width),
            'height': int(window.height),
            'transform': new_transform,
            'compress': 'lzw',
        })
        
        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(data)
    
    print(f"[✅] Sauvegardé : {output_path}")
    print(f"     Taille : {int(window.width)} × {int(window.height)} pixels")


def main():
    print("=" * 70)
    print("  RECADRAGE AUTOUR DES BIDONVILLES")
    print("=" * 70)
    
    image_path = "data/casablancasansmask.tif"
    mask_path = "data/mask_global.tif"
    
    output_dir = "data"
    image_out = os.path.join(output_dir, "image_cropped.tif")
    mask_out = os.path.join(output_dir, "mask_cropped.tif")
    
    # 1. Trouver la bounding box des bidonvilles
    bbox = find_slum_bbox(mask_path)
    if bbox is None:
        return
    
    min_x, min_y, max_x, max_y = bbox
    slum_w = max_x - min_x
    slum_h = max_y - min_y
    
    print(f"\n[INFO] Zone des bidonvilles trouvée :")
    print(f"       Position : ({min_x}, {min_y}) → ({max_x}, {max_y})")
    print(f"       Taille   : {slum_w} × {slum_h} pixels")
    
    # 2. Ajouter une marge généreuse (50% de chaque côté)
    #    pour que le modèle apprenne aussi le contexte urbain autour des bidonvilles
    margin_x = int(slum_w * 0.5)
    margin_y = int(slum_h * 0.5)
    
    with rasterio.open(image_path) as src:
        full_w = src.width
        full_h = src.height
    
    crop_x = max(0, min_x - margin_x)
    crop_y = max(0, min_y - margin_y)
    crop_x2 = min(full_w, max_x + margin_x)
    crop_y2 = min(full_h, max_y + margin_y)
    
    crop_w = crop_x2 - crop_x
    crop_h = crop_y2 - crop_y
    
    print(f"\n[INFO] Zone de recadrage (avec marge de 50%) :")
    print(f"       Position : ({crop_x}, {crop_y}) → ({crop_x2}, {crop_y2})")
    print(f"       Taille   : {crop_w} × {crop_h} pixels")
    print(f"       Réduction : {100 * (1 - (crop_w * crop_h) / (full_w * full_h)):.1f}% de l'image originale supprimée")
    
    window = Window(col_off=crop_x, row_off=crop_y, width=crop_w, height=crop_h)
    
    # 3. Recadrer l'image satellite
    print(f"\n[1/2] Recadrage de l'image satellite...")
    crop_geotiff(image_path, image_out, window)
    
    # 4. Recadrer le masque
    print(f"\n[2/2] Recadrage du masque...")
    crop_geotiff(mask_path, mask_out, window)
    
    # 5. Statistiques du nouveau masque
    with rasterio.open(mask_out) as src:
        mask_data = src.read(1)
        if mask_data.max() > 1.0:
            mask_data = mask_data / 255.0
        binary = (mask_data > 0.5).astype(np.uint8)
        slum_pct = 100.0 * np.sum(binary > 0) / binary.size
    
    print(f"\n{'=' * 70}")
    print(f"  RÉSULTAT")
    print(f"{'=' * 70}")
    print(f"  Image recadrée  : {image_out}")
    print(f"  Masque recadré  : {mask_out}")
    print(f"  Nouvelle taille : {crop_w} × {crop_h} pixels")
    print(f"  Ancien ratio bidonville : 0.026%")
    print(f"  Nouveau ratio bidonville : {slum_pct:.2f}%")
    print(f"\n  [✅] Le déséquilibre a été réduit de {slum_pct / 0.026:.0f}x !")
    print(f"\n  Prochaine étape :")
    print(f"  1. Mettez à jour config.py pour pointer vers les nouvelles images")
    print(f"  2. Relancez : rm -rf data/tiles && python prepare_data.py")
    print(f"  3. Puis : python train.py")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
