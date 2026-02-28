"""
CatBoost J3 Pipeline — Colab Version (GPU Accelerated)
======================================================
This standalone script is designed to run on a Google Colab GPU instance.
It trains an extremely optimized, heavily regularized CatBoost model, 
ideal for noisy financial data where overfitting is the main enemy.

CatBoost on GPU is incredibly fast, allowing Optuna to explore hundreds of
trials and find the perfect L2 regularization / Random Strength balance.

Outputs:
- oof_j3_catb.npy
- test_j3_catb.npy
- submission_catb_optuna.csv
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# =====================================================================
# 1. UTILS (Purged KFold)
# =====================================================================

class PurgedKFold:
    """Time-series cross validation with embargo/purging."""
    def __init__(self, n_splits=5, purge_pct=0.02):
        self.n_splits = n_splits
        self.purge_pct = purge_pct

    def split(self, X, y=None, groups=None):
        unique_times = np.sort(np.unique(groups))
        n_times = len(unique_times)
        purge_size = int(n_times * self.purge_pct)
        fold_size = n_times // self.n_splits

        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_times

            val_times = set(unique_times[val_start:val_end])

            purge_start = max(0, val_start - purge_size)
            purge_end = min(n_times, val_end + purge_size)
            excluded_times = set(unique_times[purge_start:purge_end])
            train_times = set(unique_times) - excluded_times

            train_idx = np.where(np.isin(groups, list(train_times)))[0]
            val_idx = np.where(np.isin(groups, list(val_times)))[0]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


# =====================================================================
# 2. FEATURE ENGINEERING (J3 Logic)
# =====================================================================

def build_features(X):
    """
    Simplified but powerful J3 features.
    CatBoost handles raw data well, so we focus on key non-linear interactions.
    """
    X_new = X.copy()
    
    # Lags available
    ret_cols = [c for c in X.columns if c.startswith('RET_')]
    
    # 1. Rolling Means & STDs
    for w in [5, 10, 20]:
        cols = [f'RET_{i}' for i in range(1, w + 1) if f'RET_{i}' in X.columns]
        if len(cols) == w:
            X_new[f'RET_MEAN_{w}'] = X[cols].mean(axis=1)
            X_new[f'RET_STD_{w}'] = X[cols].std(axis=1)
            
    # 2. Momentum & Sharpe Proxies (The most powerful features in our RF)
    if 'RET_1' in X.columns and 'RET_5' in X.columns:
         X_new['MOM_POINT_1_5'] = X['RET_1'] - X['RET_5']
         
    if 'RET_1' in X.columns and 'RET_STD_20' in X_new.columns:
        X_new['SHARPE_PROXY_20'] = X['RET_1'] / (X_new['RET_STD_20'] + 1e-9)
        
    if 'RET_1' in X.columns and 'RET_STD_10' in X_new.columns:
        X_new['SHARPE_PROXY_10'] = X['RET_1'] / (X_new['RET_STD_10'] + 1e-9)

    # 3. Volume Interactions
    if 'SIGNED_VOLUME_1' in X.columns and 'RET_1' in X.columns:
        X_new['RET_x_VOL_1'] = X['RET_1'] * X['SIGNED_VOLUME_1']
        
    if 'MEDIAN_DAILY_TURNOVER' in X.columns and 'RET_1' in X.columns:
        X_new['RET_x_TURNOVER_1'] = X['RET_1'] * X['MEDIAN_DAILY_TURNOVER']

    return X_new

# =====================================================================
# 3. MAIN PIPELINE
# =====================================================================

def process_data(data_dir='Data'):
    """Loads, engineers features, and imputes data."""
    print("Loading data...")
    X_train_raw = pd.read_csv(f'{data_dir}/X_train.csv')
    y_train_raw = pd.read_csv(f'{data_dir}/y_train.csv')
    X_test_raw = pd.read_csv(f'{data_dir}/X_test.csv')

    train_df = X_train_raw.merge(y_train_raw, on='ROW_ID')
    y = (train_df['target'] > 0).astype(int)

    X_train = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_raw.drop(columns=['ROW_ID'])
    test_ids = X_test_raw['ROW_ID']
    ts_groups = X_train['TS'].str.extract(r'(\d+)')[0].astype(int).values

    X_train = X_train.drop(columns=['TS'])
    if 'TS' in X_test.columns:
        X_test = X_test.drop(columns=['TS'])

    # Drop non-numeric (CatBoost can handle categorical, but to stay aligned with other models we drop string labels)
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X_train = X_train.drop(columns=non_numeric)
        X_test = X_test.drop(columns=[c for c in non_numeric if c in X_test.columns])

    print("Engineering features...")
    X_train_feat = build_features(X_train)
    X_test_feat = build_features(X_test)
    
    print("Imputing...")
    # Tree models handle NaNs natively, but mean imputation performed better in our tests
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_feat), columns=X_train_feat.columns)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_feat), columns=X_test_feat.columns)

    return X_train_imp, X_test_imp, y, ts_groups, test_ids


def run(data_dir='Data', n_trials=100):
    print("=" * 60)
    print("  CATBOOST GPU PIPELINE (ULTRA-ROBUST)")
    print("=" * 60)

    X_train_final, X_test_final, y, ts_groups, test_ids = process_data(data_dir)
    print(f"  Final features: {X_train_final.shape[1]}")

    # Subsample for Optuna to save time (150K rows keeps time structure perfectly)
    OPTUNA_SAMPLE_SIZE = 150_000
    if len(X_train_final) > OPTUNA_SAMPLE_SIZE:
        step = len(X_train_final) // OPTUNA_SAMPLE_SIZE
        optuna_idx = np.arange(0, len(X_train_final), step)[:OPTUNA_SAMPLE_SIZE]
        X_opt = X_train_final.iloc[optuna_idx].reset_index(drop=True)
        y_opt = y.iloc[optuna_idx].reset_index(drop=True)
        ts_opt = ts_groups[optuna_idx]
    else:
        X_opt, y_opt, ts_opt = X_train_final, y, ts_groups

    # --- Optuna ---
    print(f"\n[1/3] Optuna Optimization ({n_trials} trials on GPU)...")
    
    # We use CPU by default to avoid error if no GPU, but it will be much slower.
    # In Colab, make sure to change runtime to GPU!
    task_type = "GPU"
    
    # Try initializing a tiny GPU model just to check if CUDA is actually available
    try:
        _ = CatBoostClassifier(iterations=1, task_type="GPU")
        _ = _.fit(X_opt.iloc[:10], y_opt.iloc[:10], verbose=False)
        print("  [+] GPU Detected. CatBoost will fly!")
    except Exception as e:
        print("  [!] GPU not detected or failed. Falling back to CPU. (This will be slow!)")
        task_type = "CPU"


    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 500, 2000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            
            # Anti-Overfitting arsenal:
            # 1. Depth (keep it shallow!)
            'depth': trial.suggest_int('depth', 4, 8),
            
            # 2. L2 Regularization (Higher = more conservative)
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 3.0, 50.0),
            
            # 3. Random strength (Amount of randomness to add to scoring, combats noise)
            'random_strength': trial.suggest_float('random_strength', 0.1, 10.0),
            
            # 4. Bagging Temperature (Bayesian bootstrap. Higher = more aggressive sampling)
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 5.0),
            
            'task_type': task_type,
            'verbose': False,
            'random_seed': 42
        }

        pkf = PurgedKFold(n_splits=3, purge_pct=0.02)
        scores = []
        
        for tr_idx, val_idx in pkf.split(X_opt, groups=ts_opt):
            clf = CatBoostClassifier(**params)
            clf.fit(X_opt.iloc[tr_idx], y_opt.iloc[tr_idx], 
                    eval_set=(X_opt.iloc[val_idx], y_opt.iloc[val_idx]), 
                    early_stopping_rounds=100)
            
            preds = clf.predict(X_opt.iloc[val_idx])
            scores.append(accuracy_score(y_opt.iloc[val_idx], preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best Purged CV accuracy: {study.best_value:.4f}")
    bp = study.best_params
    print("  Best Parameters:")
    for key, val in bp.items():
        print(f"    {key}: {val}")

    # --- 5-Fold OOF Generation on FULL dataset ---
    print("\n[2/3] Generating 5-Fold OOF probabilities on full dataset...")
    
    final_params = bp.copy()
    final_params['task_type'] = task_type
    final_params['verbose'] = False
    final_params['random_seed'] = 42

    pkf = PurgedKFold(n_splits=5, purge_pct=0.02)
    oof_preds = np.zeros(len(X_train_final))
    test_preds_folds = np.zeros((len(X_test_final), 5))
    
    fold = 0
    for train_idx, val_idx in pkf.split(X_train_final, groups=ts_groups):
        print(f"  Training Fold {fold+1}/5...")
        clf = CatBoostClassifier(**final_params)
        
        # We explicitly don't use early stopping here on the full set to maximize learning 
        # based on the fixed number of iterations found by Optuna
        clf.fit(X_train_final.iloc[train_idx], y.iloc[train_idx])
        
        # Save OOF probs
        oof_preds[val_idx] = clf.predict_proba(X_train_final.iloc[val_idx])[:, 1]
        
        # Predict test for this fold
        test_preds_folds[:, fold] = clf.predict_proba(X_test_final)[:, 1]
        fold += 1

    # Average test predictions across folds
    test_preds_final = test_preds_folds.mean(axis=1)

    np.save('oof_j3_catb.npy', oof_preds)
    np.save('test_j3_catb.npy', test_preds_final)
    print(f"  Saved OOF probabilities -> 'oof_j3_catb.npy'")
    print(f"  Saved TEST probabilities -> 'test_j3_catb.npy'")

    # --- Train Final Model (Optional, doing calibrated submission) ---
    print("\n[3/3] Creating submission...")
    
    # We calibrate the fold-averaged probabilities directly
    target_ratio = y.mean()
    sorted_probs = np.sort(test_preds_final)[::-1]
    n_positive = int(len(test_preds_final) * target_ratio)
    threshold = sorted_probs[min(n_positive, len(sorted_probs) - 1)]
    
    binary_preds = (test_preds_final > threshold).astype(int)

    submission = pd.DataFrame({'ROW_ID': test_ids, 'prediction': binary_preds})
    output_file = 'submission_catb_optuna.csv'
    submission.to_csv(output_file, index=False)
    
    print(f"  Threshold used: {threshold:.6f}")
    print(f"  Predictions: {binary_preds.sum()} positive ({binary_preds.mean()*100:.2f}%)")
    print(f"  Saved Submission -> '{output_file}'")
    
    print("\n" + "=" * 60)
    print("FINISHED! You can now download:")
    print(" - oof_j3_catb.npy")
    print(" - test_j3_catb.npy")
    print("And add them to your rank_blending.py back home! 🚀")
    print("=" * 60)

if __name__ == "__main__":
    # In Colab, replace with: data_dir='/content/drive/MyDrive/QRT-ChallengeENS/Data'
    run(data_dir='/content/drive/MyDrive/QRT-ChallengeENS/Data', n_trials=100)
