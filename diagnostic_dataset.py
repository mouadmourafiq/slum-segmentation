"""
Diagnostic complet du dataset — Vérifie que les données sont 100% correctes.
Ne modifie RIEN. Lecture seule.
"""
import os
import glob
import numpy as np

def main():
    print("=" * 70)
    print("  DIAGNOSTIC COMPLET DU DATASET")
    print("=" * 70)

    tiles_dir = "data/tiles"
    
    # Les tuiles sont réparties dans train/, val/, test/
    img_files = []
    msk_files = []
    for split in ["train", "val", "test"]:
        img_files += sorted(glob.glob(os.path.join(tiles_dir, split, "images", "*.npy")))
        msk_files += sorted(glob.glob(os.path.join(tiles_dir, split, "masks", "*.npy")))

    print(f"\n[1] FICHIERS")
    print(f"    Images trouvées : {len(img_files)}")
    print(f"    Masques trouvés : {len(msk_files)}")
    
    if len(img_files) != len(msk_files):
        print("    [❌ ERREUR] Le nombre d'images et de masques ne correspond pas !")
        return
    else:
        print("    [✅ OK] Même nombre d'images et de masques.")

    # ── Vérification des masques ──
    print(f"\n[2] ANALYSE DES MASQUES (scan de {len(msk_files)} fichiers...)")
    
    slum_tiles = []          # Tuiles contenant au moins 1 pixel de bidonville
    empty_tiles = []         # Tuiles 100% vides (tout à zéro)
    full_tiles = []          # Tuiles 100% remplies (tout à 1)
    weird_tiles = []         # Tuiles avec des valeurs inattendues
    
    total_slum_pixels = 0
    total_bg_pixels = 0
    
    for i, msk_path in enumerate(msk_files):
        msk = np.load(msk_path)
        
        unique_vals = np.unique(msk)
        name = os.path.basename(msk_path)
        
        # Vérifier que le masque est bien binaire {0, 1}
        valid_binary = all(v in [0, 1, 0.0, 1.0] for v in unique_vals)
        if not valid_binary:
            weird_tiles.append((name, unique_vals.tolist()))
        
        slum_count = np.sum(msk > 0)
        bg_count = np.sum(msk == 0)
        total_slum_pixels += slum_count
        total_bg_pixels += bg_count
        
        if slum_count == 0:
            empty_tiles.append(name)
        elif bg_count == 0:
            full_tiles.append(name)
        else:
            slum_tiles.append((name, int(slum_count), msk.shape))
    
    print(f"\n    --- Résultats ---")
    print(f"    Tuiles avec bidonvilles  : {len(slum_tiles)}")
    print(f"    Tuiles 100% vides        : {len(empty_tiles)}")
    print(f"    Tuiles 100% remplies     : {len(full_tiles)}")
    print(f"    Tuiles avec valeurs bizarres : {len(weird_tiles)}")
    
    if len(slum_tiles) == 0:
        print("\n    [❌ CRITIQUE] AUCUNE tuile ne contient de bidonvilles !")
        print("    Le modèle n'a RIEN à apprendre. C'est la cause du problème.")
    else:
        print(f"\n    [✅ OK] {len(slum_tiles)} tuiles contiennent des bidonvilles.")
    
    if weird_tiles:
        print(f"\n    [⚠️ ATTENTION] Tuiles avec des valeurs inattendues :")
        for name, vals in weird_tiles[:5]:
            print(f"       {name} → valeurs uniques: {vals}")
    
    # ── Détails des tuiles de bidonvilles ──
    if slum_tiles:
        print(f"\n[3] DÉTAIL DES TUILES CONTENANT DES BIDONVILLES")
        print(f"    {'Nom':<25} {'Pixels bidonville':>20} {'Taille':>15} {'% bidonville':>15}")
        print(f"    {'─' * 75}")
        for name, count, shape in slum_tiles:
            total = shape[0] * shape[1]
            pct = 100.0 * count / total
            print(f"    {name:<25} {count:>20,} {str(shape):>15} {pct:>14.2f}%")
    
    # ── Statistiques globales ──
    total_all = total_slum_pixels + total_bg_pixels
    slum_pct = 100.0 * total_slum_pixels / total_all if total_all > 0 else 0
    
    print(f"\n[4] STATISTIQUES GLOBALES DU DATASET")
    print(f"    Total pixels bidonville : {total_slum_pixels:,}")
    print(f"    Total pixels fond       : {total_bg_pixels:,}")
    print(f"    Ratio bidonville        : {slum_pct:.4f}%")
    
    if slum_pct < 0.01:
        print(f"\n    [⚠️ DÉSÉQUILIBRE EXTRÊME] Les bidonvilles représentent moins de 0.01% du dataset.")
        print(f"    C'est la raison principale pour laquelle le modèle prédit 'vide' partout.")
    
    # ── Vérification des images correspondantes ──
    print(f"\n[5] VÉRIFICATION DES IMAGES SATELLITES")
    if not img_files:
        print("    [⚠️] Aucune image trouvée, vérification impossible.")
    else:
        sample_indices = [0, len(img_files)//2, len(img_files)-1]
        if slum_tiles:
            first_slum_name = slum_tiles[0][0]
            for idx, f in enumerate(img_files):
                if os.path.basename(f) == first_slum_name:
                    sample_indices.append(idx)
                    break
    
        for idx in sample_indices:
            if idx < len(img_files):
                img = np.load(img_files[idx])
                name = os.path.basename(img_files[idx])
                print(f"    {name}: shape={img.shape}, dtype={img.dtype}, "
                      f"min={img.min()}, max={img.max()}, mean={img.mean():.1f}")
    
    # ── Vérification du split_info.json ──
    import json
    split_path = os.path.join(tiles_dir, "split_info.json")
    if os.path.exists(split_path):
        with open(split_path) as f:
            split = json.load(f)
        print(f"\n[6] RÉPARTITION TRAIN/VAL/TEST")
        
        train_names = split.get("train", [])
        val_names = split.get("val", [])
        test_names = split.get("test", [])
        
        print(f"    Train : {len(train_names)}")
        print(f"    Val   : {len(val_names)}")
        print(f"    Test  : {len(test_names)}")
        
        # Compter combien de tuiles de bidonvilles sont dans chaque split
        slum_names_set = set(s[0].replace(".npy", "") for s in slum_tiles)
        
        train_slums = sum(1 for n in train_names if n in slum_names_set)
        val_slums = sum(1 for n in val_names if n in slum_names_set)
        test_slums = sum(1 for n in test_names if n in slum_names_set)
        
        print(f"\n    Tuiles bidonville dans Train : {train_slums}")
        print(f"    Tuiles bidonville dans Val   : {val_slums}")
        print(f"    Tuiles bidonville dans Test  : {test_slums}")
        
        if train_slums == 0:
            print("\n    [❌ CRITIQUE] AUCUNE tuile de bidonville dans le set d'entraînement !")
        else:
            print(f"\n    [✅ OK] {train_slums} tuiles de bidonvilles dans l'entraînement.")
    
    print("\n" + "=" * 70)
    print("  FIN DU DIAGNOSTIC")
    print("=" * 70)

if __name__ == "__main__":
    main()
