import os
import rasterio
import matplotlib.pyplot as plt
import numpy as np

# --- CONFIGURATION ---
DATA_DIR = "/Users/oussamalouat/Documents/slum_segmentation/data"
IMAGE_PATH = os.path.join(DATA_DIR, "casablanca_cropped.tif")
MASK_PATH = os.path.join(DATA_DIR, "mask_cropped.tif")

def visualize_cropped_data():
    print("⏳ Chargement des images en cours...")

    if not os.path.exists(IMAGE_PATH) or not os.path.exists(MASK_PATH):
        print("❌ Erreur : Les images découpées sont introuvables. As-tu bien lancé le script de cropping ?")
        return

    # 1. Lire l'image satellite
    with rasterio.open(IMAGE_PATH) as src_img:
        # Lire les 3 premiers canaux (RGB)
        img = src_img.read([1, 2, 3])
        # Réarranger les dimensions pour Matplotlib (de Canaux/Hauteur/Largeur vers Hauteur/Largeur/Canaux)
        img_display = np.transpose(img, (1, 2, 0))

        # Normalisation de sécurité au cas où l'image TIF ne serait pas en format standard 0-255
        if img_display.max() > 255.0:
            img_display = (img_display / img_display.max() * 255.0)
        img_display = img_display.astype(np.uint8)

    # 2. Lire le masque
    with rasterio.open(MASK_PATH) as src_mask:
        mask = src_mask.read(1)

    print("🎨 Génération de la figure...")

    # Création de l'interface graphique avec 3 panneaux
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    # Panneau 1 : Image Satellite pure
    ax1.imshow(img_display)
    ax1.set_title("1. Image Satellite (Zone Utile)")
    ax1.axis('off')

    # Panneau 2 : Masque pur
    ax2.imshow(mask, cmap='gray')
    ax2.set_title("2. Masque (Blanc = Bidonvilles)")
    ax2.axis('off')

    # Panneau 3 : Superposition (Overlay)
    ax3.imshow(img_display)
    
    # Créer un calque rouge transparent pour les bidonvilles
    # Format RGBA (Red, Green, Blue, Alpha/Transparence)
    mask_rgba = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.float32)
    mask_rgba[mask > 0] = [1.0, 0.0, 0.0, 0.5]  # Rouge avec 50% d'opacité
    
    ax3.imshow(mask_rgba)
    ax3.set_title("3. Superposition (Vérification de l'alignement)")
    ax3.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_cropped_data()