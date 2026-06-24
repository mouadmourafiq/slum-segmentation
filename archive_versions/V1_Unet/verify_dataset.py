import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def main():
    print("================================================================")
    print("  Vérification de la qualité du Dataset")
    print("================================================================")
    
    img_dir = "data/tiles/images"
    msk_dir = "data/tiles/masks"
    
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.npy")))
    msk_files = sorted(glob.glob(os.path.join(msk_dir, "*.npy")))
    
    if not img_files:
        print("[ERREUR] Aucune tuile trouvée. Attendez la fin de prepare_data.py.")
        return
        
    print(f"[INFO] Nombre total de tuiles générées (jusqu'à présent) : {len(img_files)}")
    
    slum_tiles_count = 0
    slum_tiles_paths = []
    
    # Compter combien de tuiles contiennent réellement des bidonvilles
    print("[INFO] Vérification de l'équilibre des classes... (cela peut prendre quelques secondes)")
    for msk_path in msk_files:
        msk = np.load(msk_path)
        if msk.max() > 0:
            slum_tiles_count += 1
            slum_tiles_paths.append(msk_path)
            
    print(f"\n[RÉSULTAT STATISTIQUE]")
    print(f"  -> {slum_tiles_count} tuiles contiennent des bidonvilles.")
    print(f"  -> {len(img_files) - slum_tiles_count} tuiles sont uniquement de l'arrière-plan (vide).")
    
    if slum_tiles_count == 0:
        print("\n[❌ ERREUR] Aucune tuile ne contient de bidonvilles ! Le modèle ne pourra rien apprendre.")
    else:
        print("\n[✅ OK] Des tuiles de bidonvilles ont bien été capturées !")
        
    # Visualiser quelques tuiles pour vérifier que les masques sont bien alignés avec les images
    print("\n[INFO] Génération d'une image de vérification visuelle (dataset_preview.png)...")
    
    import random
    random.seed(42)
    
    # On choisit 3 tuiles qui contiennent des bidonvilles (s'il y en a) et 2 tuiles vides
    sample_paths = slum_tiles_paths[:3] 
    bg_paths = [p for p in msk_files if p not in slum_tiles_paths]
    
    if bg_paths:
        sample_paths += random.sample(bg_paths, min(2, len(bg_paths)))
        
    if not sample_paths:
        print("[ERREUR] Pas assez de données pour générer un aperçu.")
        return

    fig, axes = plt.subplots(len(sample_paths), 2, figsize=(10, 5 * len(sample_paths)))
    
    # Gérer le cas où on a une seule image
    if len(sample_paths) == 1: 
        axes = [axes]
    
    for i, msk_path in enumerate(sample_paths):
        img_path = msk_path.replace("masks", "images")
        
        img = np.load(img_path)
        msk = np.load(msk_path)
        
        ax_img = axes[i][0]
        ax_msk = axes[i][1]
        
        ax_img.imshow(img)
        ax_img.set_title(f"Image Satellite\n({os.path.basename(img_path)})")
        ax_img.axis("off")
        
        ax_msk.imshow(msk, cmap="gray")
        has_slum = "OUI" if msk.max() > 0 else "NON"
        ax_msk.set_title(f"Masque Binaire (Cible)\nBidonville présent : {has_slum}")
        ax_msk.axis("off")
        
    plt.tight_layout()
    plt.savefig("dataset_preview.png", dpi=150)
    print("[✅ OK] L'image 'dataset_preview.png' a été sauvegardée dans le dossier du projet.")
    print("\n[CONSEIL] Ouvrez 'dataset_preview.png' : vérifiez que les taches blanches sur la colonne de droite correspondent bien à des bidonvilles sur la colonne de gauche !")

if __name__ == "__main__":
    main()
