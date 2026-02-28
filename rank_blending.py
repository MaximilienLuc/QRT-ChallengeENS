"""
Rank Blending Ensembler — FINAL META MODEL
===========================================
Combines predictions from multiple ML models by averaging their relative ranks.
This is robust to calibration differences between algorithms.

Current Ensemble:
1. RF J5 Champion (Best LB: 0.5155)
2. RF v1 (LB: 0.5106)
3. CatBoost J3 (LB: 0.5091)
4. XGBoost J3
5. LightGBM J3
"""

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import os

def rank_blend(preds_list, weights=None):
    """Blends multiple prediction arrays using rank averaging."""
    if weights is None:
        weights = [1.0 / len(preds_list)] * len(preds_list)
    weights = np.array(weights) / np.sum(weights)
    
    blended_ranks = np.zeros_like(preds_list[0], dtype=float)
    for preds, w in zip(preds_list, weights):
        ranks = rankdata(preds)
        normalized_ranks = (ranks - 1.0) / (len(preds) - 1.0)
        blended_ranks += normalized_ranks * w
    return blended_ranks

def main():
    print("=" * 60)
    print("  RANK BLENDING — MÉGA META-MODÈLE")
    print("=" * 60)

    # 1. Load all available test probabilities
    print("\n[1/3] Loading Prediction Files...")

    available_preds = []
    available_names = []

    # --- J5 RF Champion (THE BEST: LB 0.5155) ---
    path = 'test_j5_rf.npy'
    if os.path.exists(path):
        preds = np.load(path)
        available_preds.append(preds)
        available_names.append('RF_J5')
        print(f"  [+] RF_J5 (CHAMPION): {path} (shape={preds.shape}, mean={preds.mean():.4f})")

    # --- RF v1 ---
    path = 'test_j3_rf.npy'
    if os.path.exists(path):
        preds = np.load(path)
        available_preds.append(preds)
        available_names.append('RF_v1')
        print(f"  [+] RF_v1: {path} (shape={preds.shape}, mean={preds.mean():.4f})")

    # --- CatBoost ---
    path = 'J4/test_j3_catb.npy'
    if os.path.exists(path):
        preds = np.load(path)
        available_preds.append(preds)
        available_names.append('CatB_J3')
        print(f"  [+] CatB_J3: {path} (shape={preds.shape}, mean={preds.mean():.4f})")

    # --- XGB & LGBM from CSV ---
    csv_path = 'tmp_j3_test_probas.csv'
    if os.path.exists(csv_path):
        df_probas = pd.read_csv(csv_path)
        if 'xgb' in df_probas.columns:
            preds = df_probas['xgb'].values
            available_preds.append(preds)
            available_names.append('XGB_J3')
            print(f"  [+] XGB_J3: {csv_path} (shape={preds.shape}, mean={preds.mean():.4f})")
        if 'lgbm' in df_probas.columns:
            preds = df_probas['lgbm'].values
            available_preds.append(preds)
            available_names.append('LGBM_J3')
            print(f"  [+] LGBM_J3: {csv_path} (shape={preds.shape}, mean={preds.mean():.4f})")

    # --- J4 AutoEncoder ---
    for j4_path in ['J4/test_j4_pca_ae_logreg.npy', 'test_j4_pca_ae_logreg.npy']:
        if os.path.exists(j4_path):
            preds = np.load(j4_path)
            available_preds.append(preds)
            available_names.append('J4_AutoEnc')
            print(f"  [+] J4_AutoEnc: {j4_path} (shape={preds.shape}, mean={preds.mean():.4f})")
            break

    if len(available_preds) < 2:
        print("\n  [ERROR] Need at least 2 models! Exiting.")
        return

    # Sanity check: all arrays must be the same length
    lengths = [len(p) for p in available_preds]
    if len(set(lengths)) > 1:
        print(f"\n  [ERROR] Mismatched lengths: {dict(zip(available_names, lengths))}")
        return

    # 2. Blend with weights
    print(f"\n[2/3] Rank Blending {len(available_names)} models...")

    # Weights based on actual LB scores:
    # RF_J5: 0.5155 (CHAMPION) | RF_v1: 0.5106 | CatB: 0.5091 | XGB/LGBM: ~0.50
    base_weights = {
        'RF_J5':      0.40,
        'RF_v1':      0.25,
        'CatB_J3':    0.20,
        'J4_AutoEnc': 0.10,
        'XGB_J3':     0.03,
        'LGBM_J3':    0.02,
    }

    current_weights = [base_weights.get(name, 0.10) for name in available_names]
    current_weights = np.array(current_weights) / np.sum(current_weights)

    for name, w in zip(available_names, current_weights):
        print(f"  - {name}: {w:.2%}")

    final_probas = rank_blend(available_preds, weights=current_weights)

    # 3. Calibrate threshold and create submission
    print("\n[3/3] Calibrating & generating submission...")
    y_train = pd.read_csv('Data/y_train.csv')
    target_ratio = (y_train['target'] > 0).mean()

    sorted_probas = np.sort(final_probas)[::-1]
    n_positive = int(len(final_probas) * target_ratio)
    threshold = sorted_probas[min(n_positive, len(sorted_probas) - 1)]
    final_preds = (final_probas >= threshold).astype(int)

    print(f"  Target ratio: {target_ratio:.4f}")
    print(f"  Threshold: {threshold:.6f}")
    print(f"  Predictions: {final_preds.sum()} pos / {len(final_preds)} total ({final_preds.mean()*100:.2f}%)")

    test_df = pd.read_csv('Data/X_test.csv')
    submission = pd.DataFrame({
        'ROW_ID': test_df['ROW_ID'],
        'prediction': final_preds
    })

    output_file = 'submission_rank_blend.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n  ✅ Saved -> {output_file}")
    print("=" * 60)

if __name__ == "__main__":
    main()
