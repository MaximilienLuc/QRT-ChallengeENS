"""
Compare LGBM Regressor vs LGBM Classifier on the same sample.
Same features, same best params (from reg_pipeline Optuna run).
Reports: accuracy of each, % agreement between predictions.
"""

import pandas as pd
import numpy as np
import sys, os, warnings
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator, RollingStatFeatureGenerator, GroupedFeatureGenerator,
    MomentumGenerator, VolatilityRatioGenerator, InteractionGenerator,
    ShortTermInteractionGenerator
)
from pipeline.feature_selection import FeatureEvaluator

# ── Best LGBM params from Optuna regression run ──
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

def run():
    # ── Load & feature engineer (identical to reg_pipeline) ──
    print("Loading data...")
    X_train = pd.read_csv('Data/X_train_sample.csv')
    y_train = pd.read_csv('Data/y_train_sample.csv')
    df = X_train.merge(y_train, on='ROW_ID')

    y_continuous = df['target']
    y_binary = (df['target'] > 0).astype(int)
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

    # Feature selection (drift + corr)
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
    except Exception as e:
        print(f"Drift check skipped: {e}")

    high_corr = evaluator.check_correlation(X_feat, threshold=0.95)
    X_final = X_feat.drop(columns=high_corr)
    print(f"Final features: {X_final.shape[1]}")

    # ── 5-Fold CV: both models side by side ──
    print("\nRunning 5-fold CV comparison...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    oof_reg = np.full(len(y_binary), np.nan)
    oof_clf = np.full(len(y_binary), np.nan)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_final)):
        X_tr, X_val = X_final.iloc[train_idx], X_final.iloc[val_idx]
        y_tr_cont = y_continuous.iloc[train_idx]
        y_tr_bin = y_binary.iloc[train_idx]

        # Regressor → predict continuous → binarize at 0
        reg = lgb.LGBMRegressor(**BEST_PARAMS, objective='huber', verbosity=-1, random_state=42, n_jobs=-1)
        reg.fit(X_tr, y_tr_cont)
        oof_reg[val_idx] = (reg.predict(X_val) > 0).astype(int)

        # Classifier → predict proba → binarize at 0.5
        clf = lgb.LGBMClassifier(**BEST_PARAMS, objective='binary', verbosity=-1, random_state=42, n_jobs=-1)
        clf.fit(X_tr, y_tr_bin)
        oof_clf[val_idx] = (clf.predict_proba(X_val)[:, 1] > 0.5).astype(int)

        print(f"  Fold {fold+1} done")

    # ── Results ──
    acc_reg = accuracy_score(y_binary, oof_reg)
    acc_clf = accuracy_score(y_binary, oof_clf)
    agreement = np.mean(oof_reg == oof_clf) * 100

    # Detailed agreement analysis
    both_1 = np.mean((oof_reg == 1) & (oof_clf == 1)) * 100
    both_0 = np.mean((oof_reg == 0) & (oof_clf == 0)) * 100
    reg1_clf0 = np.mean((oof_reg == 1) & (oof_clf == 0)) * 100
    reg0_clf1 = np.mean((oof_reg == 0) & (oof_clf == 1)) * 100

    print("\n" + "=" * 55)
    print("  LGBM REGRESSOR vs CLASSIFIER — Same params")
    print("=" * 55)
    print(f"  Regressor accuracy:  {acc_reg:.4f}")
    print(f"  Classifier accuracy: {acc_clf:.4f}")
    print(f"  Delta:               {acc_reg - acc_clf:+.4f}")
    print(f"\n  Agreement:           {agreement:.2f}%")
    print(f"  ├─ Both predict 1:   {both_1:.2f}%")
    print(f"  ├─ Both predict 0:   {both_0:.2f}%")
    print(f"  ├─ Reg=1, Clf=0:     {reg1_clf0:.2f}%")
    print(f"  └─ Reg=0, Clf=1:     {reg0_clf1:.2f}%")

    # Who is right when they disagree?
    disagree_mask = oof_reg != oof_clf
    if disagree_mask.sum() > 0:
        reg_right = np.mean(oof_reg[disagree_mask] == y_binary.values[disagree_mask]) * 100
        clf_right = np.mean(oof_clf[disagree_mask] == y_binary.values[disagree_mask]) * 100
        print(f"\n  On disagreements ({disagree_mask.sum()} samples):")
        print(f"  ├─ Regressor correct: {reg_right:.1f}%")
        print(f"  └─ Classifier correct: {clf_right:.1f}%")
    print("=" * 55)


if __name__ == "__main__":
    run()
