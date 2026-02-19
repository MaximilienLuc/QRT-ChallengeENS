"""
Test New Features — Incremental Evaluation
---
Measures marginal accuracy gain of each new feature generator
over the baseline, using 3-fold LGBM CV.
"""

import pandas as pd
import numpy as np
import sys, os, warnings, time
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator,
    RollingStatFeatureGenerator,
    GroupedFeatureGenerator,
    MomentumGenerator,
    VolatilityRatioGenerator,
    InteractionGenerator,
    ShortTermInteractionGenerator,
    # New generators
    HigherOrderStatsGenerator,
    SignFlipGenerator,
    VolumeFlowGenerator,
    CrossLagCorrelationGenerator,
    AllocationFeatureGenerator,
)

# LGBM params (from Optuna reg run — known good params)
LGBM_PARAMS = {
    'n_estimators': 300,
    'learning_rate': 0.05,
    'num_leaves': 69,
    'max_depth': 11,
    'min_child_samples': 12,
    'subsample': 0.58,
    'colsample_bytree': 0.71,
    'reg_alpha': 0.09,
    'reg_lambda': 0.003,
    'objective': 'binary',
    'verbosity': -1,
}


def quick_cv_accuracy(X, y, n_splits=3):
    """Fast CV accuracy with LGBM."""
    clf = lgb.LGBMClassifier(**LGBM_PARAMS, random_state=42, n_jobs=-1)
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, y, cv=folds, scoring='accuracy')
    return scores.mean(), scores.std()


def build_features(X, y, generators, fit=True):
    """Generate features from a list of generators."""
    X_out = pd.DataFrame(index=X.index)
    for gen in generators:
        if fit:
            gen.fit(X, y)
        X_part = gen.transform(X)
        X_out = pd.concat([X_out, X_part], axis=1)
    X_out = X_out.loc[:, ~X_out.columns.duplicated()]
    return X_out


def run():
    print("=" * 65)
    print("  INCREMENTAL FEATURE EVALUATION")
    print("=" * 65)

    # ── Load data ──
    print("\nLoading data...")
    X_train = pd.read_csv('Data/X_train_sample.csv')
    y_train = pd.read_csv('Data/y_train_sample.csv')
    df = X_train.merge(y_train, on='ROW_ID')
    y = (df['target'] > 0).astype(int)
    X = df.drop(columns=['target', 'ROW_ID'])
    print(f"  {len(X)} samples, {X.shape[1]} raw features")

    # ── Define generators ──
    lag_features = [f'RET_{i}' for i in range(1, 21)]
    volume_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

    # Baseline generators (existing)
    baseline_gens = [
        RawFeatureGenerator(cols=lag_features + volume_features + ['MEDIAN_DAILY_TURNOVER']),
        RollingStatFeatureGenerator(cols=lag_features, windows=[5, 10, 20], operations=['mean', 'std', 'min', 'max']),
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
        VolatilityRatioGenerator(windows=[(5, 20)]),
        GroupedFeatureGenerator(group_col='GROUP', target_cols=['RET_1', 'SIGNED_VOLUME_1'], operations=['mean']),
        InteractionGenerator(),
        ShortTermInteractionGenerator(max_lag=10),
    ]

    # New generators to test (each will be tested individually)
    new_gen_batches = {
        'HigherOrderStats': [HigherOrderStatsGenerator(windows=[5, 10, 20])],
        'SignFlip':         [SignFlipGenerator(windows=[5, 10, 20])],
        'VolumeFlow':       [VolumeFlowGenerator(windows=[5, 10])],
        'CrossLagCorr':     [CrossLagCorrelationGenerator(windows=[10, 20])],
        'Allocation':       [AllocationFeatureGenerator(target_cols=['RET_1', 'SIGNED_VOLUME_1'], operations=['mean'])],
    }

    # ── Step 1: Baseline ──
    print("\n[Baseline] Generating features...")
    t0 = time.time()
    X_baseline = build_features(X, y, baseline_gens, fit=True)
    print(f"  {X_baseline.shape[1]} features ({time.time()-t0:.1f}s)")

    print("[Baseline] Evaluating...")
    base_acc, base_std = quick_cv_accuracy(X_baseline, y)
    print(f"  Accuracy: {base_acc:.4f} (+/- {base_std:.4f})")

    # ── Step 2: Test each new batch ──
    results = {'Baseline': {'acc': base_acc, 'std': base_std, 'n_feat': X_baseline.shape[1], 'delta': 0.0}}

    for name, gens in new_gen_batches.items():
        print(f"\n[+{name}] Generating features...")
        t0 = time.time()
        X_new = build_features(X, y, gens, fit=True)
        n_new = X_new.shape[1]
        X_combined = pd.concat([X_baseline, X_new], axis=1)
        X_combined = X_combined.loc[:, ~X_combined.columns.duplicated()]
        print(f"  +{n_new} new features → {X_combined.shape[1]} total ({time.time()-t0:.1f}s)")

        print(f"[+{name}] Evaluating...")
        acc, std = quick_cv_accuracy(X_combined, y)
        delta = acc - base_acc
        results[name] = {'acc': acc, 'std': std, 'n_feat': X_combined.shape[1], 'delta': delta}
        sign = "+" if delta > 0 else ""
        print(f"  Accuracy: {acc:.4f} (+/- {std:.4f}) | Delta: {sign}{delta:.4f}")

    # ── Step 3: All combined ──
    print("\n[ALL COMBINED] Generating features...")
    all_new_gens = []
    for gens in new_gen_batches.values():
        all_new_gens.extend(gens)
    X_all_new = build_features(X, y, all_new_gens, fit=True)
    X_all = pd.concat([X_baseline, X_all_new], axis=1)
    X_all = X_all.loc[:, ~X_all.columns.duplicated()]
    print(f"  {X_all.shape[1]} total features")

    print("[ALL COMBINED] Evaluating...")
    all_acc, all_std = quick_cv_accuracy(X_all, y)
    all_delta = all_acc - base_acc
    results['ALL COMBINED'] = {'acc': all_acc, 'std': all_std, 'n_feat': X_all.shape[1], 'delta': all_delta}

    # ── Step 4: Only positive contributions ──
    positive_gens = []
    positive_names = []
    for name, gens in new_gen_batches.items():
        if results[name]['delta'] > 0:
            positive_gens.extend(gens)
            positive_names.append(name)

    if positive_gens:
        print(f"\n[POSITIVE ONLY: {', '.join(positive_names)}] Evaluating...")
        X_pos_new = build_features(X, y, positive_gens, fit=True)
        X_pos = pd.concat([X_baseline, X_pos_new], axis=1)
        X_pos = X_pos.loc[:, ~X_pos.columns.duplicated()]
        pos_acc, pos_std = quick_cv_accuracy(X_pos, y)
        pos_delta = pos_acc - base_acc
        results['POSITIVE ONLY'] = {'acc': pos_acc, 'std': pos_std, 'n_feat': X_pos.shape[1], 'delta': pos_delta}

    # ── Summary ──
    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  {'Generator':<22} {'#Feat':>6} {'Accuracy':>10} {'Delta':>8}")
    print(f"  {'-'*22} {'-'*6} {'-'*10} {'-'*8}")
    for name, r in results.items():
        sign = "+" if r['delta'] > 0 else ""
        marker = " ✓" if r['delta'] > 0.001 else (" ~" if r['delta'] > -0.001 else " ✗")
        if name == 'Baseline':
            marker = " ──"
        print(f"  {name:<22} {r['n_feat']:>6} {r['acc']:>10.4f} {sign}{r['delta']:>7.4f}{marker}")

    print("=" * 65)


if __name__ == "__main__":
    run()
