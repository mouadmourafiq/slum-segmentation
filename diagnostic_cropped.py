"""
Diagnostic complet des images recadrées + génération de visualisations.
Compare l'AVANT (image originale) et l'APRÈS (image recadrée).
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Pas besoin d'écran
import matplotlib.pyplot as plt
import rasterio
from rasterio.windows import Window
from rasterio.enums import Resampling

def scan_mask(mask_path, block_size=1024):
    """Scanne un masque par blocs et retourne les statistiques + un aperçu miniature."""
    with rasterio.open(mask_path) as src:
        h, w = src.height, src.width
        total = h * w
        
        # Aperçu miniature
        scale = min(1.0, 800 / max(h, w))
        mini_h, mini_w = max(1, int(h * scale)), max(1, int(w * scale))
        mini = np.zeros((mini_h, mini_w), dtype=np.uint8)
        
        slum_px = 0
        
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                rh = min(block_size, h - y)
                rw = min(block_size, w - x)
                window = Window(col_off=x, row_off=y, width=rw, height=rh)
                block = src.read(1, window=window)
                
                if block.max() > 1.0:
                    block = block / 255.0
                binary = (block > 0.5).astype(np.uint8)
                
                count = int(np.sum(binary > 0))
                slum_px += count
                
                if count > 0:
                    sy = min(int(y * scale), mini_h - 1)
                    sx = min(int(x * scale), mini_w - 1)
                    ey = min(sy + max(1, int(rh * scale)), mini_h)
                    ex = min(sx + max(1, int(rw * scale)), mini_w)
                    mini[sy:ey, sx:ex] = 1
    
    bg_px = total - slum_px
    pct = 100.0 * slum_px / total if total > 0 else 0
    return slum_px, bg_px, pct, mini, w, h


def load_rgb_preview(image_path, max_size=800):
    """Charge un aperçu RGB réduit d'un GeoTIFF."""
    with rasterio.open(image_path) as src:
        h, w = src.height, src.width
        scale = min(1.0, max_size / max(h, w))
        out_h, out_w = max(1, int(h * scale)), max(1, int(w * scale))
        
        # Lire les 3 premières bandes en version réduite
        rgb = src.read(
            indexes=[1, 2, 3],
            out_shape=(3, out_h, out_w),
            resampling=Resampling.nearest
        )
        rgb = np.transpose(rgb, (1, 2, 0))  # (H, W, 3)
        return rgb


def main():
    print("=" * 70)
    print("  DIAGNOSTIC COMPLET — AVANT vs APRÈS RECADRAGE")
    print("=" * 70)
    
    # Chemins
    orig_image = "data/casablancasansmask.tif"
    orig_mask  = "data/mask_global.tif"
    crop_image = "data/image_cropped.tif"
    crop_mask  = "data/mask_cropped.tif"
    
    # ── 1. Scan de l'image ORIGINALE ──
    print("\n[1/2] Analyse de l'image ORIGINALE...")
    orig_slum, orig_bg, orig_pct, orig_mini, orig_w, orig_h = scan_mask(orig_mask)
    print(f"       Taille : {orig_w} × {orig_h}")
    print(f"       Bidonvilles : {orig_slum:,} pixels ({orig_pct:.4f}%)")
    
    # ── 2. Scan de l'image RECADRÉE ──
    print("\n[2/2] Analyse de l'image RECADRÉE...")
    crop_slum, crop_bg, crop_pct, crop_mini, crop_w, crop_h = scan_mask(crop_mask)
    print(f"       Taille : {crop_w} × {crop_h}")
    print(f"       Bidonvilles : {crop_slum:,} pixels ({crop_pct:.4f}%)")
    
    # ── 3. Comparaison ──
    improvement = crop_pct / orig_pct if orig_pct > 0 else 0
    size_reduction = 100 * (1 - (crop_w * crop_h) / (orig_w * orig_h))
    
    print(f"\n{'=' * 70}")
    print(f"  COMPARAISON")
    print(f"{'=' * 70}")
    print(f"  {'Métrique':<30} {'Originale':>15} {'Recadrée':>15}")
    print(f"  {'─' * 60}")
    print(f"  {'Taille':<30} {f'{orig_w}×{orig_h}':>15} {f'{crop_w}×{crop_h}':>15}")
    print(f"  {'Pixels totaux':<30} {orig_w*orig_h:>15,} {crop_w*crop_h:>15,}")
    print(f"  {'Pixels bidonville':<30} {orig_slum:>15,} {crop_slum:>15,}")
    print(f"  {'Ratio bidonville':<30} {orig_pct:>14.4f}% {crop_pct:>14.4f}%")
    print(f"  {'Amélioration du ratio':<30} {'':>15} {f'{improvement:.0f}x':>15}")
    print(f"  {'Réduction de taille':<30} {'':>15} {f'{size_reduction:.1f}%':>15}")
    
    # ── 4. Générer la visualisation ──
    print(f"\n[INFO] Génération de l'image de diagnostic (diagnostic_cropped.png)...")
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle("Diagnostic Complet — AVANT vs APRÈS Recadrage", fontsize=16, fontweight='bold')
    
    # ── Ligne 1 : AVANT (Image originale) ──
    # Pie chart AVANT
    axes[0][0].pie(
        [orig_bg, orig_slum],
        labels=['Arrière-plan', 'Bidonvilles'],
        colors=['#cccccc', '#ff6b6b'],
        autopct='%1.3f%%',
        startangle=140,
        textprops={'fontsize': 11}
    )
    axes[0][0].set_title('AVANT — Répartition des Classes\n(Image Originale)', fontweight='bold')
    
    # Carte thermique AVANT
    axes[0][1].imshow(orig_mini, cmap='inferno', aspect='auto')
    axes[0][1].set_title(f'AVANT — Emplacement des bidonvilles\n({orig_w}×{orig_h} px)', fontweight='bold')
    axes[0][1].axis('off')
    
    # Aperçu satellite AVANT
    try:
        orig_rgb = load_rgb_preview(orig_image)
        axes[0][2].imshow(orig_rgb)
        axes[0][2].set_title('AVANT — Vue Satellite\n(Image Originale)', fontweight='bold')
    except Exception as e:
        axes[0][2].text(0.5, 0.5, f'Erreur: {e}', ha='center', va='center')
        axes[0][2].set_title('AVANT — Vue Satellite')
    axes[0][2].axis('off')
    
    # ── Ligne 2 : APRÈS (Image recadrée) ──
    # Pie chart APRÈS
    axes[1][0].pie(
        [crop_bg, crop_slum],
        labels=['Arrière-plan', 'Bidonvilles'],
        colors=['#cccccc', '#ff6b6b'],
        autopct='%1.3f%%',
        startangle=140,
        textprops={'fontsize': 11}
    )
    axes[1][0].set_title('APRÈS — Répartition des Classes\n(Image Recadrée)', fontweight='bold')
    
    # Carte thermique APRÈS
    axes[1][1].imshow(crop_mini, cmap='inferno', aspect='auto')
    axes[1][1].set_title(f'APRÈS — Emplacement des bidonvilles\n({crop_w}×{crop_h} px)', fontweight='bold')
    axes[1][1].axis('off')
    
    # Aperçu satellite APRÈS
    try:
        crop_rgb = load_rgb_preview(crop_image)
        axes[1][2].imshow(crop_rgb)
        axes[1][2].set_title('APRÈS — Vue Satellite\n(Image Recadrée)', fontweight='bold')
    except Exception as e:
        axes[1][2].text(0.5, 0.5, f'Erreur: {e}', ha='center', va='center')
        axes[1][2].set_title('APRÈS — Vue Satellite')
    axes[1][2].axis('off')
    
    plt.tight_layout()
    plt.savefig('diagnostic_cropped.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("[✅] Image sauvegardée : diagnostic_cropped.png")
    print("\n[CONSEIL] Ouvrez 'diagnostic_cropped.png' pour voir la comparaison visuelle AVANT/APRÈS !")
    print("=" * 70)


if __name__ == "__main__":
    main()
