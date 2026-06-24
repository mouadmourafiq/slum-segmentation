import os
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from rasterio.enums import Resampling

def main():
    print("================================================================")
    print("  Visualisation Rapide des Données Brutes (Sans Tiling)")
    print("================================================================")
    
    mask_path = "data/mask_global.tif"
    
    if not os.path.exists(mask_path):
        print(f"[ERREUR] Le fichier {mask_path} n'existe pas.")
        return
        
    with rasterio.open(mask_path) as src:
        width = src.width
        height = src.height
        total_pixels = width * height
        
        print(f"[INFO] Image originale : {width} x {height} pixels ({total_pixels:,} pixels au total)")
        
        # Pour ne pas faire crasher la RAM et être très rapide (sans affecter le script de Tiling),
        # nous allons charger une version réduite de la carte (2.5% de la taille d'origine).
        # On utilise Resampling.max pour être sûr de ne rater aucun pixel de bidonville (point blanc) 
        # même en dézoomant massivement.
        scale = 0.025
        out_shape = (int(height * scale), int(width * scale))
        
        print(f"[INFO] Scan rapide de l'image (cela prend environ 15-20 secondes)...")
        
        # On crée une image vide pour la carte thermique
        mask_small = np.zeros(out_shape, dtype=np.uint8)
        
        slum_pixels = 0
        bg_pixels = 0
        
        # On lit par blocs de 1024x1024 pour aller très vite sans surcharger la RAM
        from rasterio.windows import Window
        import math
        
        block_size = 1024
        
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                read_w = min(block_size, width - x)
                read_h = min(block_size, height - y)
                
                window = Window(col_off=x, row_off=y, width=read_w, height=read_h)
                block_data = src.read(1, window=window)
                
                if block_data.max() > 1.0:
                    block_data = block_data / 255.0
                block_binary = (block_data > 0.5).astype(np.uint8)
                
                # Comptage exact
                slum_count = int(np.sum(block_binary == 1))
                bg_count = int(np.sum(block_binary == 0))
                
                slum_pixels += slum_count
                bg_pixels += bg_count
                
                # Si le bloc contient un bidonville, on allume le pixel correspondant dans l'aperçu
                if slum_count > 0:
                    small_y = min(int(y * scale), out_shape[0] - 1)
                    small_x = min(int(x * scale), out_shape[1] - 1)
                    # On allume un petit carré de 2x2 pixels sur la carte pour que ce soit bien visible
                    mask_small[max(0, small_y-1):small_y+2, max(0, small_x-1):small_x+2] = 1
                    
    # Statistiques exactes calculées !
    slum_pct = (slum_pixels / total_pixels) * 100
    
    print("\n[STATISTIQUES ESTIMÉES SUR LA CARTE GLOBALE]")
    print(f" -> Pourcentage d'arrière-plan (Vide) : {100 - slum_pct:.3f} %")
    print(f" -> Pourcentage de bidonvilles (Cible) : {slum_pct:.3f} %")
    
    # Création du graphique
    print("\n[INFO] Création de l'image de visualisation (raw_stats.png)...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Pie chart
    labels = ['Arrière-plan (Vide)', 'Bidonvilles (Cible)']
    sizes = [bg_pixels, slum_pixels]
    colors = ['#cccccc', '#ff9999']
    
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.3f%%', startangle=140)
    axes[0].set_title('Répartition des Classes (Extrême Déséquilibre)')
    
    # 2. Carte thermique (Où sont les bidonvilles ?)
    # Utilisation d'un fond noir avec les bidonvilles en jaune fluo pour bien les voir
    axes[1].imshow(mask_small, cmap='inferno')
    axes[1].set_title('Emplacement des bidonvilles sur la carte géante')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('raw_stats.png', dpi=150, bbox_inches='tight')
    print("[✅ OK] L'image 'raw_stats.png' a été sauvegardée dans le dossier de votre projet.")
    print("Vous pouvez l'ouvrir pour voir à quel point les bidonvilles sont rares sur l'image géante !")

if __name__ == "__main__":
    main()
