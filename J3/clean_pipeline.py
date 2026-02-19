"""
Clean Pipeline — Optuna + Purged KFold + Submission
---
Fixes:
- Removed InteractionGenerator (redundant with ShortTermInteraction)
- Removed GroupedFeatureGenerator (temporal leakage via group stats)
- Drift threshold 0.60 (consistent with sample evaluation)
- Optuna anti-overfit search space + Purged KFold
- Calibrated threshold for ~50% positive predictions
- Generates submission at the end
"""

import pandas as pd
import numpy as np
import sys, os, warnings
import lightgbm as lgb
import optuna
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator, RollingStatFeatureGenerator,
    MomentumGenerator, VolatilityRatioGenerator,
    ShortTermInteractionGenerator,
)
from pipeline.feature_selection import FeatureEvaluator

GENERATORS_FN = lambda: [
    RawFeatureGenerator(
        cols=[f'RET_{i}' for i in range(1, 21)]
           + [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
           + ['MEDIAN_DAILY_TURNOVER']
    ),
    RollingStatFeatureGenerator(
        cols=[f'RET_{i}' for i in range(1, 21)],
        windows=[5, 10, 20],
        operations=['mean', 'std', 'min', 'max']
    ),
    MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
    VolatilityRatioGenerator(windows=[(5, 20)]),
    ShortTermInteractionGenerator(max_lag=10),
]


class PurgedKFold:
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


def build_features(X, y=None, generators=None, fit=True):
    X_feat = pd.DataFrame(index=X.index)
    for gen in generators:
        if fit and hasattr(gen, 'fit'):
            gen.fit(X, y)
        X_part = gen.transform(X)
        X_feat = pd.concat([X_feat, X_part], axis=1)
    X_feat = X_feat.loc[:, ~X_feat.columns.duplicated()]
    return X_feat


def purged_cv_score(params, X, y, ts_groups, n_splits=3):
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=0.02)
    val_scores, train_scores = [], []
    for train_idx, val_idx in pkf.split(X, groups=ts_groups):
        model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        val_scores.append(accuracy_score(y.iloc[val_idx], model.predict(X.iloc[val_idx])))
        train_scores.append(accuracy_score(y.iloc[train_idx], model.predict(X.iloc[train_idx])))
    return np.mean(val_scores), np.std(val_scores), np.mean(train_scores)


def run():
    print("=" * 65)
    print("  CLEAN PIPELINE — OPTUNA + PURGED KFOLD")
    print("=" * 65)

    # ── 1. Load ──
    print("\n[1/6] Loading data...")
    X_train_full = pd.read_csv('Data/X_train.csv')
    y_train_full = pd.read_csv('Data/y_train.csv')
    X_test_full = pd.read_csv('Data/X_test.csv')

    train_df = X_train_full.merge(y_train_full, on='ROW_ID')
    y_full = (train_df['target'] > 0).astype(int)
    X_full = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_full.drop(columns=['ROW_ID'])
    ts_full = X_full['TS'].str.extract(r'(\d+)')[0].astype(int).values

    print(f"  Train: {X_full.shape}, Test: {X_test.shape}")

    # ── 2. Feature Engineering ──
    print("\n[2/6] Feature engineering (no InteractionGen, no GroupedFeature)...")
    generators = GENERATORS_FN()
    X_feat = build_features(X_full, y_full, generators, fit=True)
    X_test_feat = build_features(X_test, generators=generators, fit=False)
    print(f"  {X_feat.shape[1]} features generated")

    # ── 3. Feature Selection (drift 0.60) ──
    print("\n[3/6] Feature selection (drift threshold 0.60)...")
    evaluator = FeatureEvaluator(cv=3)
    drifting = evaluator.check_drift(X_feat, X_test_feat, threshold=0.60)
    X_feat = X_feat.drop(columns=drifting)
    X_test_feat = X_test_feat.drop(columns=drifting)
    print(f"  Dropped {len(drifting)} drifting features")

    high_corr = evaluator.check_correlation(X_feat, threshold=0.95)
    X_feat = X_feat.drop(columns=high_corr)
    X_test_feat = X_test_feat.drop(columns=high_corr)
    print(f"  Dropped {len(high_corr)} correlated features")
    print(f"  Final: {X_feat.shape[1]} features")

    # ── 4. Optuna ──
    print("\n[4/6] Optuna optimization (30 trials, 3-fold Purged KFold)...")

    def objective(trial):
        params = {
            'objective': 'binary', 'verbosity': -1, 'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 300),
            'subsample': trial.suggest_float('subsample', 0.4, 0.8),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        }
        val_acc, _, train_acc = purged_cv_score(params, X_feat, y_full, ts_full, n_splits=3)
        trial.set_user_attr('train_acc', train_acc)
        trial.set_user_attr('gap', train_acc - val_acc)
        return val_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best Purged CV accuracy: {study.best_value:.4f}")
    print(f"  Train-val gap: {study.best_trial.user_attrs['gap']:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ── 5. Final evaluation (5-fold) ──
    print("\n[5/6] Final 5-fold Purged KFold evaluation...")
    best_p = study.best_params.copy()
    best_p.update({'objective': 'binary', 'verbosity': -1, 'boosting_type': 'gbdt'})
    val_acc, val_std, train_acc = purged_cv_score(best_p, X_feat, y_full, ts_full, n_splits=5)
    print(f"  Train: {train_acc:.4f} | Val: {val_acc:.4f} (+/- {val_std:.4f}) | Gap: {train_acc-val_acc:.4f}")

    # ── 6. Train final model + calibrated submission ──
    print("\n[6/6] Training final model & generating submission...")
    model = lgb.LGBMClassifier(**best_p, random_state=42, n_jobs=-1)
    model.fit(X_feat, y_full)
    probs = model.predict_proba(X_test_feat)[:, 1]

    # Calibrate threshold to match training positive ratio
    target_ratio = y_full.mean()
    sorted_probs = np.sort(probs)[::-1]
    n_positive = int(len(probs) * target_ratio)
    threshold = sorted_probs[n_positive]
    preds = (probs > threshold).astype(int)

    print(f"  Threshold: {threshold:.6f} (target ratio: {target_ratio:.4f})")
    print(f"  Predictions: {preds.sum()} pos / {len(preds)} total ({preds.mean()*100:.1f}%)")

    submission = pd.DataFrame({'ROW_ID': X_test_full['ROW_ID'], 'score': preds})
    output_file = 'submission_clean_purged.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file}")

    # Top 5 trials recap
    print("\n─── Top 5 Trials ───")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False).head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number']):>2}: val={row['value']:.4f}, gap={row['user_attrs_gap']:.4f}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    run()
