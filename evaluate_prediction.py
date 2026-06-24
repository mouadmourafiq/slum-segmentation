import os
import rasterio
import numpy as np

def calculate_metrics(y_true, y_pred):
    """Calcule l'IoU, la Précision, le Rappel et le F1-Score (Dice)."""
    # Vrais Positifs (le modèle a dit bidonville, et c'est bien un bidonville)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    
    # Faux Positifs (le modèle a dit bidonville, mais c'est du vide en réalité)
    fp = np.sum((y_true == 0) & (y_pred == 1))
    
    # Faux Négatifs (le modèle a dit vide, mais c'était un bidonville en réalité)
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    union = tp + fp + fn
    
    iou = tp / union if union > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return iou, precision, recall, f1, tp, fp, fn

def main():
    print("=" * 60)
    print("  ÉVALUATION GLOBALE DE LA CARTE PRODUITE")
    print("=" * 60)
    
    import config
    pred_path = config.PREDICTION_OUTPUT_PATH
    true_path = config.INPUT_MASK_PATH
    
    if not os.path.exists(pred_path) or not os.path.exists(true_path):
        print("[❌ ERREUR] Fichiers introuvables. Avez-vous lancé l'inférence ?")
        return
    
    print(f"[INFO] Comparaison de :\n   Prédiction : {pred_path}\n   Vérité     : {true_path}\n")
    
    with rasterio.open(pred_path) as src:
        pred_mask = src.read(1)
        pred_mask = (pred_mask > 0).astype(np.uint8)
        
    with rasterio.open(true_path) as src:
        true_mask = src.read(1)
        if true_mask.max() > 1.0:
            true_mask = true_mask / 255.0
        true_mask = (true_mask > 0.5).astype(np.uint8)
        
    print("[INFO] Calcul mathématique pixel par pixel en cours...\n")
    iou, precision, recall, f1, tp, fp, fn = calculate_metrics(true_mask, pred_mask)
    
    print("-" * 60)
    print(f"  🏆 F1-Score (Dice)               : {f1:.4f}  <-- Métrique Principale")
    print(f"  🎯 IoU (Intersection over Union) : {iou:.4f}")
    print("-" * 60)
    print(f"  Précision (Qualité des alertes) : {precision:.4f} (Combien de vrais bidonvilles parmi tout ce qu'il a détecté ?)")
    print(f"  Rappel (Sensibilité)            : {recall:.4f} (Combien de vrais bidonvilles a-t-il réussi à trouver ?)")
    print("-" * 60)
    
    print("\n[ANALYSE]")
    if precision < 0.3:
        print("-> Le modèle fait beaucoup de FAUX POSITIFS (il est trop paranoïaque).")
    if recall < 0.3:
        print("-> Le modèle rate beaucoup de bidonvilles (il est trop aveugle).")
        
    print("\n[RAPPEL] L'entraînement actuel n'a duré que 3 Époques ! Mettez NUM_EPOCHS=20 dans config.py pour améliorer drastiquement ces scores.")
    print("=" * 60)

if __name__ == "__main__":
    main()
