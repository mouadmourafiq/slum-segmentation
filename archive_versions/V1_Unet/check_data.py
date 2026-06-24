import os
import glob
import numpy as np
import config

def check_dataset():
    search_path = os.path.join(config.TRAIN_DIR, "masks", "*.npy")
    print("\n🔍 DIAGNOSTIC DES CHEMINS")
    print(f"Chemin de recherche : {search_path}")
    
    # Vérification de l'existence des dossiers
    print(f"Le dossier TRAIN_DIR existe-t-il ? : {os.path.exists(config.TRAIN_DIR)}")
    if os.path.exists(config.TRAIN_DIR):
        print(f"Contenu de {config.TRAIN_DIR} : {os.listdir(config.TRAIN_DIR)}")
    
    mask_paths = sorted(glob.glob(search_path))
    
    if not mask_paths:
        print("\n❌ Aucun masque trouvé ! Il y a un problème avec les dossiers ci-dessus.")
        return

    total_tiles = len(mask_paths)
    empty_tiles = 0
    total_pixels = 0
    slum_pixels = 0

    print(f"\n✅ {total_tiles} masques trouvés. Analyse en cours...")
    
    for path in mask_paths:
        mask = np.load(path)
        pixels_in_tile = mask.size
        slums_in_tile = np.sum(mask > 0)
        
        total_pixels += pixels_in_tile
        slum_pixels += slums_in_tile
        
        if slums_in_tile == 0:
            empty_tiles += 1

    print("\n=== 📊 RÉSULTATS DE TA DATA ===")
    print(f"Total des tuiles : {total_tiles}")
    print(f"Tuiles 100% VIDES (aucun bidonville) : {empty_tiles} (soit {empty_tiles/total_tiles*100:.1f}%)")
    print(f"Tuiles UTILES : {total_tiles - empty_tiles}")
    print("-" * 30)
    print(f"Total des pixels analysés : {total_pixels:,}")
    if total_pixels > 0:
        print(f"Total des pixels 'bidonville' : {slum_pixels:,} (soit {slum_pixels/total_pixels*100:.4f}%)")

if __name__ == "__main__":
    check_dataset()