"""
Random Forest V2 Submission Pipeline — J3
=========================================
Improvements over v1:
1. New targeted features: Return acceleration (RET_DIFF), multi-scale Sharpe & Momentum
2. PurgedKFold used inside Optuna to strictly avoid time-leakage during tuning
3. Tuning set increased to 150,000 samples for better generalization
4. max_features forced to smaller values (0.05 - 0.2) to massively decorrelate trees 
   and combat the noise/overfitting causing the CV-LB gap.
"""

import pandas as pd
import numpy as np
import optuna
import sys
import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator,
    RollingStatFeatureGenerator,
    MomentumGenerator,
    VolatilityRatioGenerator,
    InteractionGenerator,
    ShortTermInteractionGenerator,
    ReturnDifferenceGenerator
)
from pipeline.feature_selection import FeatureEvaluator

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

def build_features(X, generators, fitted=False, y=None):
    X_out = pd.DataFrame(index=X.index)
    for gen in generators:
        if not fitted:
            gen.fit(X, y)
        X_part = gen.transform(X)
        X_out = pd.concat([X_out, X_part], axis=1)
    return X_out.loc[:, ~X_out.columns.duplicated()]

def run():
    print("=" * 60)
    print("  Random Forest V2 Pipeline (J3) ")
    print("=" * 60)

    # 1. Load Data
    print("\n[1/7] Loading Data...")
    X_train_raw = pd.read_csv('Data/X_train.csv')
    y_train_raw = pd.read_csv('Data/y_train.csv')
    X_test_raw = pd.read_csv('Data/X_test.csv')

    train_df = X_train_raw.merge(y_train_raw, on='ROW_ID')
    y = (train_df['target'] > 0).astype(int)
    X = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_raw.drop(columns=['ROW_ID'])
    
    # Extract TS groups for PurgedKFold
    ts_groups = X["TS"].str.extract(r"(\d+)")[0].astype(int).values

    print(f"  Train: {X.shape}  |  Test: {X_test.shape}")

    # 2. Feature Engineering
    print("\n[2/7] Feature Engineering...")
    lag_features = [f'RET_{i}' for i in range(1, 21)]
    volume_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

    generators = [
        RawFeatureGenerator(cols=lag_features + volume_features + ['MEDIAN_DAILY_TURNOVER']),
        RollingStatFeatureGenerator(cols=lag_features, windows=[5, 10, 20], operations=['mean', 'std', 'min', 'max']),
        MomentumGenerator(windows=[(1, 5), (1, 10), (1, 20), (5, 20)]),
        VolatilityRatioGenerator(windows=[(5, 10), (5, 20)]),
        ReturnDifferenceGenerator(lags=[(1, 5), (1, 10), (1, 20), (5, 10), (5, 20)]),
        InteractionGenerator(),
        ShortTermInteractionGenerator(max_lag=10),
    ]

    X_train_feat = build_features(X, generators, fitted=False, y=y)
    X_test_feat = build_features(X_test, generators, fitted=True)

    print(f"  Features generated: {X_train_feat.shape[1]}")

    # 3. Mean Imputation
    print("\n[3/7] Mean Imputation...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train_feat), columns=X_train_feat.columns, index=X_train_feat.index)
    X_test_imp = pd.DataFrame(imputer.transform(X_test_feat), columns=X_test_feat.columns, index=X_test_feat.index)

    # 4. Adversarial Validation & Correlation Filter
    print("\n[4/7] Filtering Drift & Correlation...")
    evaluator = FeatureEvaluator(cv=3)
    drifting = evaluator.check_drift(X_train_imp, X_test_imp, threshold=0.60)
    print(f"  Dropping {len(drifting)} drifting features.")
    
    X_train_imp = X_train_imp.drop(columns=drifting)
    X_test_imp = X_test_imp.drop(columns=drifting)

    high_corr = evaluator.check_correlation(X_train_imp, threshold=0.95)
    print(f"  Dropping {len(high_corr)} highly correlated features.")
    
    X_final = X_train_imp.drop(columns=high_corr)
    X_test_final = X_test_imp.drop(columns=high_corr)
    print(f"  Final feature count: {X_final.shape[1]}")

    # 5. Optuna Hyperparameter Tuning
    print("\n[5/7] Optuna Optimization (PurgedKFold on 150_000 samples)...")
    OPTUNA_SAMPLE_SIZE = 150_000
    
    # Subsample linearly over time to preserve time structure
    step = len(X_final) // OPTUNA_SAMPLE_SIZE
    optuna_idx = np.arange(0, len(X_final), step)[:OPTUNA_SAMPLE_SIZE]
    
    X_opt = X_final.iloc[optuna_idx].reset_index(drop=True)
    y_opt = y.iloc[optuna_idx].reset_index(drop=True)
    ts_opt = ts_groups[optuna_idx]

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 250, 600),
            'max_depth': trial.suggest_int('max_depth', 8, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 10, 60),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 30),
            # STRICT LIMIT on max_features to force tree decorrelation
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.05, 0.1, 0.15]),
            'random_state': 42,
            'n_jobs': -1,
        }
        
        pkf_optuna = PurgedKFold(n_splits=3, purge_pct=0.02)
        scores = []
        for tr_idx, val_idx in pkf_optuna.split(X_opt, groups=ts_opt):
            clf = RandomForestClassifier(**params)
            clf.fit(X_opt.iloc[tr_idx], y_opt.iloc[tr_idx])
            preds = clf.predict(X_opt.iloc[val_idx])
            scores.append(accuracy_score(y_opt.iloc[val_idx], preds))
            
        return np.mean(scores)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=15, show_progress_bar=True)

    print(f"\n  Best Purged CV Accuracy (on subsample): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # 6. Train Final Model on FULL DATASET
    print("\n[6/7] Training Final Model on 100% of data...")
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    final_rf = RandomForestClassifier(**best_params)
    final_rf.fit(X_final, y)
    
    # Save the model
    os.makedirs('models', exist_ok=True)
    joblib.dump(final_rf, 'models/rf_v2_best.pkl')

    # 7. Generate Submission
    print("\n[7/7] Generating Submission...")
    preds = final_rf.predict(X_test_final)

    submission = pd.DataFrame({
        'ROW_ID': X_test_raw['ROW_ID'],
        'prediction': preds,
    })

    output_file = 'submission_rf_v2_optuna.csv'
    submission.to_csv(output_file, index=False)
    print(f"  Submission saved to {output_file}")

    # Log feature importances
    importances = pd.DataFrame({
        'feature': X_final.columns,
        'importance': final_rf.feature_importances_,
    }).sort_values('importance', ascending=False)
    
    print("\n  Top 20 Feature Importances:")
    print(importances.head(20).to_string(index=False))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    run()
