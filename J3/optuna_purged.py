"""
Optuna Optimization with Purged K-Fold
---
Optimizes LGBM hyperparameters using time-aware Purged KFold CV
to get realistic (non-leaky) accuracy estimates.
Search space biased toward anti-overfitting (stronger regularization).
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
    RawFeatureGenerator, RollingStatFeatureGenerator, GroupedFeatureGenerator,
    MomentumGenerator, VolatilityRatioGenerator, InteractionGenerator,
    ShortTermInteractionGenerator,
)
from pipeline.feature_selection import FeatureEvaluator


class PurgedKFold:
    """Time-aware K-Fold with purge gap between train and val."""
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


def purged_cv_score(params, X, y, ts_groups, n_splits=5):
    """Run Purged KFold CV and return mean val accuracy + train accuracy."""
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=0.02)
    val_scores = []
    train_scores = []

    for train_idx, val_idx in pkf.split(X, groups=ts_groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
        model.fit(X_tr, y_tr)

        val_scores.append(accuracy_score(y_val, model.predict(X_val)))
        train_scores.append(accuracy_score(y_tr, model.predict(X_tr)))

    return np.mean(val_scores), np.std(val_scores), np.mean(train_scores)


def run():
    print("=" * 70)
    print("  OPTUNA + PURGED K-FOLD OPTIMIZATION")
    print("=" * 70)

    # ── Load & Feature Engineer ──
    print("\nLoading data...")
    X_train = pd.read_csv('Data/X_train_sample.csv')
    y_train = pd.read_csv('Data/y_train_sample.csv')
    df = X_train.merge(y_train, on='ROW_ID')

    y = (df['target'] > 0).astype(int)
    X = df.drop(columns=['target', 'ROW_ID'])
    ts_num = X['TS'].str.extract(r'(\d+)')[0].astype(int).values

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

    print("Feature engineering...")
    X_feat = pd.DataFrame(index=X.index)
    for gen in generators:
        gen.fit(X, y)
        X_feat = pd.concat([X_feat, gen.transform(X)], axis=1)
    X_feat = X_feat.loc[:, ~X_feat.columns.duplicated()]

    # Feature selection
    print("Feature selection (drift + correlation)...")
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
    X_final = X_feat.drop(columns=high_corr)
    print(f"  Final: {X_final.shape[1]} features")

    # ── Baseline with old params ──
    print("\n─── Baseline (old params) ───")
    old_params = {
        'n_estimators': 300, 'learning_rate': 0.05, 'num_leaves': 69,
        'max_depth': 11, 'min_child_samples': 12, 'subsample': 0.58,
        'colsample_bytree': 0.71, 'reg_alpha': 0.09, 'reg_lambda': 0.003,
        'objective': 'binary', 'verbosity': -1,
    }
    val_acc, val_std, train_acc = purged_cv_score(old_params, X_final, y, ts_num)
    print(f"  Train: {train_acc:.4f} | Val: {val_acc:.4f} (+/- {val_std:.4f}) | Gap: {train_acc-val_acc:.4f}")
    baseline_val = val_acc

    # ── Optuna with Anti-Overfit search space ──
    print("\n─── Optuna Optimization (30 trials, anti-overfit focus) ───")

    def objective(trial):
        params = {
            'objective': 'binary',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            # Anti-overfit: smaller trees, more regularization
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 64),           # was 20-150, now capped at 64
            'max_depth': trial.suggest_int('max_depth', 2, 8),              # was -1 to 12, now max 8
            'min_child_samples': trial.suggest_int('min_child_samples', 30, 200),  # was 5-100, now min 30
            'subsample': trial.suggest_float('subsample', 0.4, 0.8),        # was 0.5-1.0, cap at 0.8
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.7),  # more aggressive feature sampling
            'reg_alpha': trial.suggest_float('reg_alpha', 0.1, 50.0, log=True),    # much stronger L1
            'reg_lambda': trial.suggest_float('reg_lambda', 0.1, 50.0, log=True),  # much stronger L2
            'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),     # require minimum info gain
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 5),
        }

        val_acc, _, train_acc = purged_cv_score(params, X_final, y, ts_num, n_splits=3)

        # Log the gap for analysis
        trial.set_user_attr('train_acc', train_acc)
        trial.set_user_attr('gap', train_acc - val_acc)

        return val_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30, show_progress_bar=True)

    # ── Results ──
    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best Purged CV accuracy: {study.best_value:.4f} (baseline: {baseline_val:.4f}, delta: {study.best_value-baseline_val:+.4f})")
    print(f"  Best gap: {study.best_trial.user_attrs['gap']:.4f}")
    print(f"\n  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ── Re-evaluate best with 5-fold ──
    print("\n─── Final evaluation (5-fold Purged) ───")
    best_p = study.best_params.copy()
    best_p.update({'objective': 'binary', 'verbosity': -1, 'boosting_type': 'gbdt'})
    val_acc, val_std, train_acc = purged_cv_score(best_p, X_final, y, ts_num, n_splits=5)
    print(f"  Train: {train_acc:.4f} | Val: {val_acc:.4f} (+/- {val_std:.4f}) | Gap: {train_acc-val_acc:.4f}")

    # ── Top 5 trials ──
    print("\n─── Top 5 Trials ───")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False).head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number']):>2}: val={row['value']:.4f}, train={row['user_attrs_train_acc']:.4f}, gap={row['user_attrs_gap']:.4f}")

    print("\n" + "=" * 70)
    print(f"  Baseline (old params, purged):  {baseline_val:.4f}")
    print(f"  Best (optimized, purged):       {val_acc:.4f} ({val_acc-baseline_val:+.4f})")
    print("=" * 70)


if __name__ == "__main__":
    run()
