"""
Full Submission — Anti-Overfit LGBM
---
Uses the best params from Optuna + Purged KFold optimization
trained on the FULL dataset. Generates submission CSV.
"""

import pandas as pd
import numpy as np
import sys, os, warnings
import lightgbm as lgb

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator, RollingStatFeatureGenerator, GroupedFeatureGenerator,
    MomentumGenerator, VolatilityRatioGenerator, InteractionGenerator,
    ShortTermInteractionGenerator,
)
from pipeline.feature_selection import FeatureEvaluator

# Best params from Optuna + Purged KFold (trial #28)
BEST_PARAMS = {
    'objective': 'binary',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'n_estimators': 223,
    'learning_rate': 0.01096,
    'num_leaves': 19,
    'max_depth': 3,
    'min_child_samples': 182,
    'subsample': 0.624,
    'colsample_bytree': 0.335,
    'reg_alpha': 0.395,
    'reg_lambda': 0.755,
    'min_split_gain': 0.078,
    'bagging_freq': 3,
}


def run():
    print("=" * 60)
    print("  FULL SUBMISSION — Anti-Overfit LGBM")
    print("=" * 60)

    # ── 1. Load Full Data ──
    print("\n[1/5] Loading full data...")
    X_train_full = pd.read_csv('Data/X_train.csv')
    y_train_full = pd.read_csv('Data/y_train.csv')
    X_test_full = pd.read_csv('Data/X_test.csv')

    train_df = X_train_full.merge(y_train_full, on='ROW_ID')
    y = (train_df['target'] > 0).astype(int)
    X = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_full.drop(columns=['ROW_ID'])

    print(f"  Train: {X.shape}")
    print(f"  Test:  {X_test.shape}")

    # ── 2. Feature Engineering ──
    print("\n[2/5] Feature engineering...")
    lag_features = [f'RET_{i}' for i in range(1, 21)]
    volume_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

    generators = [
        RawFeatureGenerator(cols=lag_features + volume_features + ['MEDIAN_DAILY_TURNOVER']),
        RollingStatFeatureGenerator(cols=lag_features, windows=[5, 10, 20], operations=['mean', 'std', 'min', 'max']),
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
        VolatilityRatioGenerator(windows=[(5, 20)]),
        GroupedFeatureGenerator(group_col='GROUP', target_cols=['RET_1', 'SIGNED_VOLUME_1'], operations=['mean']),
        InteractionGenerator(),
        ShortTermInteractionGenerator(max_lag=10),
    ]

    print("  Generating train features...")
    X_feat = pd.DataFrame(index=X.index)
    for gen in generators:
        gen.fit(X, y)
        X_feat = pd.concat([X_feat, gen.transform(X)], axis=1)
    X_feat = X_feat.loc[:, ~X_feat.columns.duplicated()]

    print("  Generating test features...")
    X_test_feat = pd.DataFrame(index=X_test.index)
    for gen in generators:
        X_part = gen.transform(X_test)
        X_test_feat = pd.concat([X_test_feat, X_part], axis=1)
    X_test_feat = X_test_feat.loc[:, ~X_test_feat.columns.duplicated()]

    print(f"  Generated: {X_feat.shape[1]} features")

    # ── 3. Feature Selection ──
    print("\n[3/5] Feature selection...")
    evaluator = FeatureEvaluator(cv=3)

    # Drift check
    drifting = evaluator.check_drift(X_feat, X_test_feat, threshold=0.70)
    print(f"  Dropping {len(drifting)} drifting features")
    X_feat = X_feat.drop(columns=drifting)
    X_test_feat = X_test_feat.drop(columns=drifting)

    # Correlation check
    high_corr = evaluator.check_correlation(X_feat, threshold=0.95)
    print(f"  Dropping {len(high_corr)} highly correlated features")
    X_final = X_feat.drop(columns=high_corr)
    X_test_final = X_test_feat.drop(columns=high_corr)
    print(f"  Final: {X_final.shape[1]} features")

    # ── 4. Train Model ──
    print("\n[4/5] Training final model...")
    model = lgb.LGBMClassifier(**BEST_PARAMS, random_state=42, n_jobs=-1)
    model.fit(X_final, y)
    print("  Model trained.")

    # ── 5. Predict & Submit ──
    print("\n[5/5] Generating submission...")
    probs = model.predict_proba(X_test_final)[:, 1]
    preds = (probs > 0.5).astype(int)

    submission = pd.DataFrame({
        'ROW_ID': X_test_full['ROW_ID'],
        'score': preds,
    })

    output_file = 'submission_lgbm_purged_antioverfit.csv'
    submission.to_csv(output_file, index=False)

    print(f"\n  Predictions: {preds.sum()} positives / {len(preds)} total ({preds.mean()*100:.1f}%)")
    print(f"  Submission saved to {output_file}")
    print("=" * 60)


if __name__ == "__main__":
    run()
