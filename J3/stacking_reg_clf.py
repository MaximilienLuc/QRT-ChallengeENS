"""
Stacking: LGBM Regressor + LGBM Classifier
---
Base models produce OOF meta-features, then a Logistic Regression meta-learner
combines them. Also tests simple vote and average for comparison.
"""

import pandas as pd
import numpy as np
import sys, os, warnings
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator, RollingStatFeatureGenerator, GroupedFeatureGenerator,
    MomentumGenerator, VolatilityRatioGenerator, InteractionGenerator,
    ShortTermInteractionGenerator
)
from pipeline.feature_selection import FeatureEvaluator

# Best params from Optuna run
BEST_PARAMS = {
    'n_estimators': 455,
    'learning_rate': 0.0592,
    'num_leaves': 69,
    'max_depth': 11,
    'min_child_samples': 12,
    'subsample': 0.579,
    'colsample_bytree': 0.713,
    'reg_alpha': 0.092,
    'reg_lambda': 0.003,
}


def prepare_data():
    """Load data, feature engineer, feature select — identical to previous scripts."""
    print("Loading data...")
    X_train = pd.read_csv('Data/X_train_sample.csv')
    y_train = pd.read_csv('Data/y_train_sample.csv')
    df = X_train.merge(y_train, on='ROW_ID')

    y_continuous = df['target'].values
    y_binary = (df['target'] > 0).astype(int).values
    X = df.drop(columns=['target', 'ROW_ID'])

    lag_features = [f'RET_{i}' for i in range(1, 21)]
    volume_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
    generators = [
        RawFeatureGenerator(cols=lag_features + volume_features + ['MEDIAN_DAILY_TURNOVER']),
        RollingStatFeatureGenerator(cols=lag_features, windows=[5, 10, 20], operations=['mean', 'std', 'min', 'max']),
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
        VolatilityRatioGenerator(windows=[(5, 20)]),
        GroupedFeatureGenerator(group_col='GROUP', target_cols=['RET_1', 'SIGNED_VOLUME_1'], operations=['mean']),
        InteractionGenerator(),
        ShortTermInteractionGenerator(max_lag=10)
    ]

    print("Feature engineering...")
    X_feat = pd.DataFrame(index=X.index)
    for gen in generators:
        gen.fit(X, y_binary)
        X_feat = pd.concat([X_feat, gen.transform(X)], axis=1)
    X_feat = X_feat.loc[:, ~X_feat.columns.duplicated()]

    print("Feature selection...")
    evaluator = FeatureEvaluator(cv=3)
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
        X_feat = X_feat.drop(columns=drifting)
        print(f"  Dropped {len(drifting)} drifting features")
    except Exception as e:
        print(f"  Drift check skipped: {e}")

    high_corr = evaluator.check_correlation(X_feat, threshold=0.95)
    X_feat = X_feat.drop(columns=high_corr)
    print(f"  Dropped {len(high_corr)} correlated features")
    print(f"  Final: {X_feat.shape[1]} features, {len(y_binary)} samples")

    return X_feat, y_continuous, y_binary


def run():
    X_final, y_continuous, y_binary = prepare_data()

    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ── Step 1: Generate OOF meta-features from base models ──
    print(f"\n[Stacking] Generating OOF predictions ({n_splits}-fold)...")

    oof_reg_raw = np.zeros(len(y_binary))     # Raw regression output
    oof_clf_proba = np.zeros(len(y_binary))   # Classifier probability

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_final)):
        X_tr, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
        y_tr_cont = y_continuous[train_idx]
        y_tr_bin = y_binary[train_idx]

        # Base model 1: LGBM Regressor (Huber)
        reg = lgb.LGBMRegressor(**BEST_PARAMS, objective='huber', verbosity=-1, random_state=42, n_jobs=-1)
        reg.fit(X_tr, y_tr_cont)
        oof_reg_raw[val_idx] = reg.predict(X_val)

        # Base model 2: LGBM Classifier
        clf = lgb.LGBMClassifier(**BEST_PARAMS, objective='binary', verbosity=-1, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr_bin)
        oof_clf_proba[val_idx] = clf.predict_proba(X_val)[:, 1]

        print(f"  Fold {fold+1}/{n_splits} done")

    # ── Step 2: Build meta-features matrix ──
    meta_features = np.column_stack([oof_reg_raw, oof_clf_proba])
    print(f"\nMeta-features shape: {meta_features.shape}")

    # ── Step 3: Evaluate ensemble strategies ──
    print("\n" + "=" * 60)
    print("  ENSEMBLE STRATEGIES COMPARISON")
    print("=" * 60)

    results = {}

    # 3a. Individual baselines
    acc_reg = accuracy_score(y_binary, (oof_reg_raw > 0).astype(int))
    acc_clf = accuracy_score(y_binary, (oof_clf_proba > 0.5).astype(int))
    results['Regressor (alone)'] = acc_reg
    results['Classifier (alone)'] = acc_clf

    # 3b. Simple average of proba + reg (need to normalize reg to [0,1] range)
    # Normalize reg: shift so that 0 maps to ~0.5
    from sklearn.preprocessing import MinMaxScaler
    reg_norm = (oof_reg_raw - oof_reg_raw.min()) / (oof_reg_raw.max() - oof_reg_raw.min())
    avg = (reg_norm + oof_clf_proba) / 2
    acc_avg = accuracy_score(y_binary, (avg > 0.5).astype(int))
    results['Simple Average'] = acc_avg

    # 3c. Majority vote
    pred_reg = (oof_reg_raw > 0).astype(int)
    pred_clf = (oof_clf_proba > 0.5).astype(int)
    vote = pred_reg + pred_clf  # 0, 1, or 2
    # Tie-break: use classifier proba
    pred_vote = np.where(vote >= 1, 1, 0)  # Majority = at least 1 out of 2
    # Actually with 2 models, majority vote = OR. Let's do strict majority (both agree)
    pred_vote_strict = np.where(vote == 2, 1, np.where(vote == 0, 0, pred_clf))  # tie-break → clf
    acc_vote = accuracy_score(y_binary, pred_vote_strict)
    results['Majority Vote (tie→clf)'] = acc_vote

    # 3d. Weighted vote (try different weights)
    best_w_acc = 0
    best_w = 0
    for w in np.arange(0.0, 1.05, 0.05):
        blended = w * reg_norm + (1 - w) * oof_clf_proba
        acc_w = accuracy_score(y_binary, (blended > 0.5).astype(int))
        if acc_w > best_w_acc:
            best_w_acc = acc_w
            best_w = w
    results[f'Weighted Avg (w_reg={best_w:.2f})'] = best_w_acc

    # 3e. Logistic Regression stacking (CV to avoid overfitting)
    stacking_kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    stacking_preds = np.zeros(len(y_binary))

    for train_idx, val_idx in stacking_kf.split(meta_features, y_binary):
        meta_tr = meta_features[train_idx]
        meta_val = meta_features[val_idx]
        y_tr = y_binary[train_idx]

        lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        lr.fit(meta_tr, y_tr)
        stacking_preds[val_idx] = lr.predict(meta_val)

    acc_stack = accuracy_score(y_binary, stacking_preds)
    results['Stacking (LogReg)'] = acc_stack

    # 3f. Stacking with more regularization variants
    for C in [0.01, 0.1, 10.0]:
        stacking_preds_c = np.zeros(len(y_binary))
        for train_idx, val_idx in stacking_kf.split(meta_features, y_binary):
            lr = LogisticRegression(C=C, max_iter=1000, random_state=42)
            lr.fit(meta_features[train_idx], y_binary[train_idx])
            stacking_preds_c[val_idx] = lr.predict(meta_features[val_idx])
        acc_c = accuracy_score(y_binary, stacking_preds_c)
        results[f'Stacking (LogReg C={C})'] = acc_c

    # ── Print results ──
    print(f"\n  {'Strategy':<32} {'Accuracy':>10}")
    print(f"  {'-'*32} {'-'*10}")
    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ← best" if acc == max(results.values()) else ""
        print(f"  {name:<32} {acc:>10.4f}{marker}")

    best_name = max(results, key=results.get)
    print(f"\n  🏆 Best: {best_name} ({results[best_name]:.4f})")
    print("=" * 60)

    # Show LogReg weights
    lr_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_full.fit(meta_features, y_binary)
    print(f"\n  LogReg stacking weights: reg={lr_full.coef_[0][0]:.4f}, clf={lr_full.coef_[0][1]:.4f}")
    print(f"  Intercept: {lr_full.intercept_[0]:.4f}")


if __name__ == "__main__":
    run()
