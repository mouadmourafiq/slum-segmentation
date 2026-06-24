import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling

def main():
    print("=" * 60)
    print("  GÉNÉRATION DE LA CARTE FINALE EN IMAGE")
    print("=" * 60)
    
    import config
    img_path = config.INPUT_IMAGE_PATH
    pred_path = config.PREDICTION_OUTPUT_PATH
    
    # On charge une version réduite pour ne pas faire exploser la RAM
    max_size = 3000
    
    print("[INFO] Chargement de l'image satellite...")
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_size / max(h, w))
        out_h, out_w = int(h * scale), int(w * scale)
        
        rgb = src.read(
            [1, 2, 3],
            out_shape=(3, out_h, out_w),
            resampling=Resampling.average
        )
        rgb = np.transpose(rgb, (1, 2, 0))
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
            
    print("[INFO] Chargement des prédictions de l'IA...")
    with rasterio.open(pred_path) as src:
        mask = src.read(
            1,
            out_shape=(out_h, out_w),
            resampling=Resampling.nearest
        )
        mask = (mask > 0).astype(np.float32)
        
    print("[INFO] Création de l'image (fusion)...")
    fig, ax = plt.subplots(figsize=(20, 12))
    
    # Afficher la vue satellite
    ax.imshow(rgb)
    
    # Ajouter les bidonvilles en Rouge fluo semi-transparent
    red_overlay = np.zeros((*mask.shape, 4))
    red_overlay[mask == 1] = [1, 0, 0, 0.6]  # Rouge avec 60% d'opacité
    
    ax.imshow(red_overlay)
    ax.set_title("Carte générée par l'IA : Détection des Bidonvilles (en rouge)", fontsize=18, fontweight='bold')
    ax.axis('off')
    
    out_file = "output/final_map_preview.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, bbox_inches='tight', facecolor='black')
    plt.close()
    
    print(f"\n[✅] C'est prêt ! Image sauvegardée : {out_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
