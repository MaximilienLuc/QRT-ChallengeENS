"""
Regression Pipeline — QRT Challenge
---
Same feature processing as run_pipeline.py, but uses regression models
instead of classification. Predictions are binarized (1 if pred > threshold, 0 otherwise)
with threshold optimized via CV.

Models: Linear Regression, Random Forest Regressor (Optuna), LightGBM Regressor (Optuna)
"""

import pandas as pd
import numpy as np
import sys
import os
import warnings
import lightgbm as lgb
import optuna
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from scipy.optimize import minimize_scalar

# Silence optuna logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

sys.path.append(os.path.abspath('.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator,
    RollingStatFeatureGenerator,
    GroupedFeatureGenerator,
    MomentumGenerator,
    VolatilityRatioGenerator,
    InteractionGenerator,
    ShortTermInteractionGenerator
)
from pipeline.feature_selection import FeatureEvaluator


# ─────────────────────────────────────────────
# Utility: Optimal threshold search
# ─────────────────────────────────────────────
def find_optimal_threshold(y_true_binary, y_pred_continuous, low=-0.05, high=0.05):
    """
    Find threshold t that maximizes accuracy of (y_pred > t) vs y_true_binary.
    Searches in [low, high] range.
    """
    def neg_accuracy(t):
        preds = (y_pred_continuous > t).astype(int)
        return -accuracy_score(y_true_binary, preds)

    result = minimize_scalar(neg_accuracy, bounds=(low, high), method='bounded')
    best_t = result.x
    best_acc = -result.fun
    return best_t, best_acc


# ─────────────────────────────────────────────
# Feature Engineering (same as run_pipeline.py)
# ─────────────────────────────────────────────
def build_features(X, y_for_fit=None, generators=None, fit=False):
    """Generate features. If fit=True, calls fit+transform; else just transform."""
    X_out = pd.DataFrame(index=X.index)
    for gen in generators:
        if fit:
            gen.fit(X, y_for_fit)
        X_part = gen.transform(X)
        X_out = pd.concat([X_out, X_part], axis=1)
    # Remove duplicate columns
    X_out = X_out.loc[:, ~X_out.columns.duplicated()]
    return X_out


def get_generators():
    lag_features = [f'RET_{i}' for i in range(1, 21)]
    volume_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
    return [
        RawFeatureGenerator(cols=lag_features + volume_features + ['MEDIAN_DAILY_TURNOVER']),
        RollingStatFeatureGenerator(cols=lag_features, windows=[5, 10, 20], operations=['mean', 'std', 'min', 'max']),
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
        VolatilityRatioGenerator(windows=[(5, 20)]),
        GroupedFeatureGenerator(group_col='GROUP', target_cols=['RET_1', 'SIGNED_VOLUME_1'], operations=['mean']),
        InteractionGenerator(),
        ShortTermInteractionGenerator(max_lag=10)
    ]


# ─────────────────────────────────────────────
# Cross-Validated Regression → Binary Accuracy
# ─────────────────────────────────────────────
def cv_regression_accuracy(model_fn, X, y_continuous, y_binary, n_splits=5):
    """
    K-Fold CV: train regression on y_continuous, optimize threshold on OOF,
    report accuracy vs y_binary.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_preds = np.full(len(y_continuous), np.nan)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr = y_continuous.iloc[train_idx]

        model = model_fn()
        model.fit(X_tr, y_tr)
        oof_preds[val_idx] = model.predict(X_val)

    # Optimize threshold on all OOF predictions
    best_t, best_acc = find_optimal_threshold(y_binary.values, oof_preds)
    return best_acc, best_t, oof_preds


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def run():
    print("=" * 60)
    print("  REGRESSION PIPELINE — QRT Challenge")
    print("=" * 60)

    # ── 1. Load Data ──
    print("\n[1/5] Loading sample data...")
    try:
        X_train = pd.read_csv('Data/X_train_sample.csv')
        y_train = pd.read_csv('Data/y_train_sample.csv')
    except FileNotFoundError:
        print("ERROR: Sample data files not found in Data/")
        return

    df = X_train.merge(y_train, on='ROW_ID')
    y_continuous = df['target']                        # Regression target
    y_binary = (df['target'] > 0).astype(int)          # For accuracy eval
    X = df.drop(columns=['target', 'ROW_ID'])

    print(f"   Samples: {len(X)} | Raw features: {X.shape[1]}")
    print(f"   Target distribution: {y_binary.value_counts().to_dict()}")

    # ── 2. Feature Engineering ──
    print("\n[2/5] Feature engineering...")
    generators = get_generators()
    X_feat = build_features(X, y_for_fit=y_binary, generators=generators, fit=True)
    print(f"   Generated {X_feat.shape[1]} features")

    # ── 3. Feature Selection (drift + correlation) ──
    print("\n[3/5] Feature selection...")
    evaluator = FeatureEvaluator(cv=3)

    # Drift check
    try:
        X_test_real = pd.read_csv('Data/X_test.csv')
        X_test_sample = X_test_real.sample(n=min(len(X), 10000), random_state=42).drop(columns=['ROW_ID']).reset_index(drop=True)

        X_test_feat = pd.DataFrame(index=X_test_sample.index)
        for gen in generators:
            X_part = gen.transform(X_test_sample).reset_index(drop=True)
            X_test_feat = X_test_feat.reset_index(drop=True)
            X_test_feat = pd.concat([X_test_feat, X_part], axis=1)
        X_test_feat = X_test_feat.loc[:, ~X_test_feat.columns.duplicated()]

        drifting = evaluator.check_drift(X_feat, X_test_feat, threshold=0.60)
        print(f"   Dropping {len(drifting)} drifting features")
        X_feat = X_feat.drop(columns=drifting)
    except Exception as e:
        print(f"   Drift check skipped: {e}")

    # Correlation check
    high_corr = evaluator.check_correlation(X_feat, threshold=0.95)
    print(f"   Dropping {len(high_corr)} highly correlated features")
    X_final = X_feat.drop(columns=high_corr)
    print(f"   Final feature count: {X_final.shape[1]}")

    # Fill NaNs for models that can't handle them (Ridge, RF)
    X_final_filled = X_final.fillna(0)

    results = {}

    # ── 4a. Linear Regression (Ridge) ──
    print("\n[4/5] Training models...")
    print("\n   ── Ridge Regression ──")
    acc, threshold, _ = cv_regression_accuracy(
        model_fn=lambda: Ridge(alpha=1.0),
        X=X_final_filled, y_continuous=y_continuous, y_binary=y_binary
    )
    results['Ridge'] = {'accuracy': acc, 'threshold': threshold}
    print(f"   Accuracy: {acc:.4f} | Optimal threshold: {threshold:.5f}")

    # ── 4b. Random Forest Regressor (light Optuna) ──
    print("\n   ── Random Forest Regressor (Optuna) ──")

    def rf_objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 20, 80),
            'max_features': trial.suggest_float('max_features', 0.3, 0.7),
        }
        acc, _, _ = cv_regression_accuracy(
            model_fn=lambda p=params: RandomForestRegressor(**p, random_state=42, n_jobs=-1),
            X=X_final_filled, y_continuous=y_continuous, y_binary=y_binary,
            n_splits=3  # 3-fold for speed
        )
        return acc

    rf_study = optuna.create_study(direction='maximize')
    rf_study.optimize(rf_objective, n_trials=5, show_progress_bar=False)

    # Re-evaluate best RF with 5-fold
    best_rf_params = rf_study.best_params
    acc, threshold, _ = cv_regression_accuracy(
        model_fn=lambda: RandomForestRegressor(**best_rf_params, random_state=42, n_jobs=-1),
        X=X_final_filled, y_continuous=y_continuous, y_binary=y_binary,
        n_splits=5
    )
    results['RF'] = {'accuracy': acc, 'threshold': threshold, 'params': best_rf_params}
    print(f"   Best params: {best_rf_params}")
    print(f"   Accuracy: {acc:.4f} | Optimal threshold: {threshold:.5f}")

    # ── 4c. LightGBM Regressor (light Optuna) ──
    print("\n   ── LightGBM Regressor (Optuna) ──")

    def lgbm_objective(trial):
        params = {
            'objective': 'huber',
            'verbosity': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 80),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        }
        acc, _, _ = cv_regression_accuracy(
            model_fn=lambda p=params: lgb.LGBMRegressor(**p, random_state=42, n_jobs=-1),
            X=X_final, y_continuous=y_continuous, y_binary=y_binary,
            n_splits=3
        )
        return acc

    lgbm_study = optuna.create_study(direction='maximize')
    lgbm_study.optimize(lgbm_objective, n_trials=10, show_progress_bar=False)

    # Re-evaluate best LGBM with 5-fold
    best_lgbm_params = lgbm_study.best_params
    best_lgbm_params.update({'objective': 'huber', 'verbosity': -1})
    acc, threshold, _ = cv_regression_accuracy(
        model_fn=lambda: lgb.LGBMRegressor(**best_lgbm_params, random_state=42, n_jobs=-1),
        X=X_final, y_continuous=y_continuous, y_binary=y_binary,
        n_splits=5
    )
    results['LGBM'] = {'accuracy': acc, 'threshold': threshold, 'params': best_lgbm_params}
    print(f"   Best params: {best_lgbm_params}")
    print(f"   Accuracy: {acc:.4f} | Optimal threshold: {threshold:.5f}")

    # ── 5. Summary ──
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<12} {'Accuracy':>10} {'Threshold':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*12}")
    for name, res in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"  {name:<12} {res['accuracy']:>10.4f} {res['threshold']:>12.5f}")

    best_model = max(results, key=lambda k: results[k]['accuracy'])
    print(f"\n  🏆 Best: {best_model} ({results[best_model]['accuracy']:.4f})")
    print("=" * 60)


if __name__ == "__main__":
    run()
