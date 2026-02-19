"""
Purged K-Fold vs Standard K-Fold Comparison
---
Compares standard shuffled K-Fold (data leakage risk) vs time-based
Purged K-Fold (no future data leakage) to see if overfitting to
temporal patterns inflates CV scores.
"""

import pandas as pd
import numpy as np
import sys, os, warnings
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

warnings.filterwarnings('ignore')
sys.path.append(os.path.abspath('.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator, RollingStatFeatureGenerator, GroupedFeatureGenerator,
    MomentumGenerator, VolatilityRatioGenerator, InteractionGenerator,
    ShortTermInteractionGenerator,
)

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


class PurgedKFold:
    """
    Time-aware K-Fold that:
    1. Splits by time (no shuffling)
    2. Adds a 'purge' gap between train and validation to prevent leakage
    """
    def __init__(self, n_splits=5, purge_pct=0.02):
        self.n_splits = n_splits
        self.purge_pct = purge_pct

    def split(self, X, y=None, groups=None):
        """
        groups: array of time indices (e.g. TS_NUM) for each sample.
        Splits time into n_splits contiguous blocks.
        Train = all blocks before validation block.
        Purge gap = purge_pct of total dates removed between train and val.
        """
        unique_times = np.sort(np.unique(groups))
        n_times = len(unique_times)
        purge_size = int(n_times * self.purge_pct)

        fold_size = n_times // self.n_splits

        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_times

            val_times = set(unique_times[val_start:val_end])

            # Train: everything NOT in val and NOT in purge zone
            purge_start = max(0, val_start - purge_size)
            purge_end = min(n_times, val_end + purge_size)
            excluded_times = set(unique_times[purge_start:purge_end])

            train_times = set(unique_times) - excluded_times

            train_idx = np.where(np.isin(groups, list(train_times)))[0]
            val_idx = np.where(np.isin(groups, list(val_times)))[0]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

    def get_n_splits(self):
        return self.n_splits


class TimeSeriesPurgedKFold:
    """
    Walk-forward / expanding window variant:
    Train always comes BEFORE validation in time.
    Purge gap between train and val.
    """
    def __init__(self, n_splits=5, purge_pct=0.02):
        self.n_splits = n_splits
        self.purge_pct = purge_pct

    def split(self, X, y=None, groups=None):
        unique_times = np.sort(np.unique(groups))
        n_times = len(unique_times)
        purge_size = int(n_times * self.purge_pct)

        # Reserve first 30% as minimum training set
        min_train_end = int(n_times * 0.3)
        remaining = n_times - min_train_end
        fold_size = remaining // self.n_splits

        for i in range(self.n_splits):
            val_start_idx = min_train_end + i * fold_size
            val_end_idx = min_train_end + (i + 1) * fold_size if i < self.n_splits - 1 else n_times

            # Train: everything before val, minus purge zone
            train_end_idx = max(0, val_start_idx - purge_size)
            train_times = set(unique_times[:train_end_idx])
            val_times = set(unique_times[val_start_idx:val_end_idx])

            train_idx = np.where(np.isin(groups, list(train_times)))[0]
            val_idx = np.where(np.isin(groups, list(val_times)))[0]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx

    def get_n_splits(self):
        return self.n_splits


def cv_accuracy(model_fn, X, y, cv, groups=None):
    """Run CV and return per-fold + mean accuracy."""
    scores = []
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups=groups)):
        model = model_fn()
        X_tr = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
        X_val = X.iloc[val_idx] if hasattr(X, 'iloc') else X[val_idx]
        y_tr = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        model.fit(X_tr, y_tr)

        # Train accuracy (for overfitting check)
        train_acc = accuracy_score(y_tr, model.predict(X_tr))
        val_acc = accuracy_score(y_val, model.predict(X_val))
        scores.append({'fold': fold+1, 'train_acc': train_acc, 'val_acc': val_acc,
                       'train_size': len(train_idx), 'val_size': len(val_idx)})

    return scores


def run():
    print("=" * 70)
    print("  PURGED K-FOLD vs STANDARD K-FOLD")
    print("=" * 70)

    # ── Load & Feature Engineer ──
    print("\nLoading data...")
    X_train = pd.read_csv('Data/X_train_sample.csv')
    y_train = pd.read_csv('Data/y_train_sample.csv')
    df = X_train.merge(y_train, on='ROW_ID')

    y = (df['target'] > 0).astype(int)
    X = df.drop(columns=['target', 'ROW_ID'])

    # Extract time index
    ts_num = X['TS'].str.extract(r'(\d+)')[0].astype(int).values
    print(f"  {len(X)} samples, {len(np.unique(ts_num))} unique dates")

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
    print(f"  {X_feat.shape[1]} features")

    model_fn = lambda: lgb.LGBMClassifier(**LGBM_PARAMS, random_state=42, n_jobs=-1)

    # ── 1. Standard Shuffled K-Fold ──
    print("\n─── Standard Shuffled K-Fold (5-fold) ───")
    std_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    std_scores = cv_accuracy(model_fn, X_feat, y, std_cv)

    for s in std_scores:
        gap = s['train_acc'] - s['val_acc']
        print(f"  Fold {s['fold']}: train={s['train_acc']:.4f} val={s['val_acc']:.4f} gap={gap:.4f} (n_train={s['train_size']}, n_val={s['val_size']})")

    std_val_mean = np.mean([s['val_acc'] for s in std_scores])
    std_train_mean = np.mean([s['train_acc'] for s in std_scores])
    print(f"  MEAN: train={std_train_mean:.4f} val={std_val_mean:.4f} gap={std_train_mean-std_val_mean:.4f}")

    # ── 2. Purged K-Fold (contiguous blocks, purge gap) ──
    print("\n─── Purged K-Fold (5-fold, 2% purge gap) ───")
    purged_cv = PurgedKFold(n_splits=5, purge_pct=0.02)
    purged_scores = cv_accuracy(model_fn, X_feat, y, purged_cv, groups=ts_num)

    for s in purged_scores:
        gap = s['train_acc'] - s['val_acc']
        print(f"  Fold {s['fold']}: train={s['train_acc']:.4f} val={s['val_acc']:.4f} gap={gap:.4f} (n_train={s['train_size']}, n_val={s['val_size']})")

    purged_val_mean = np.mean([s['val_acc'] for s in purged_scores])
    purged_train_mean = np.mean([s['train_acc'] for s in purged_scores])
    print(f"  MEAN: train={purged_train_mean:.4f} val={purged_val_mean:.4f} gap={purged_train_mean-purged_val_mean:.4f}")

    # ── 3. Walk-Forward (expanding window, purge gap) ──
    print("\n─── Walk-Forward Purged (5-fold, 2% purge) ───")
    wf_cv = TimeSeriesPurgedKFold(n_splits=5, purge_pct=0.02)
    wf_scores = cv_accuracy(model_fn, X_feat, y, wf_cv, groups=ts_num)

    for s in wf_scores:
        gap = s['train_acc'] - s['val_acc']
        print(f"  Fold {s['fold']}: train={s['train_acc']:.4f} val={s['val_acc']:.4f} gap={gap:.4f} (n_train={s['train_size']}, n_val={s['val_size']})")

    wf_val_mean = np.mean([s['val_acc'] for s in wf_scores])
    wf_train_mean = np.mean([s['train_acc'] for s in wf_scores])
    print(f"  MEAN: train={wf_train_mean:.4f} val={wf_val_mean:.4f} gap={wf_train_mean-wf_val_mean:.4f}")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  COMPARISON SUMMARY")
    print("=" * 70)
    print(f"  {'Method':<30} {'Train':>8} {'Val':>8} {'Gap':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'Standard Shuffled KFold':<30} {std_train_mean:>8.4f} {std_val_mean:>8.4f} {std_train_mean-std_val_mean:>8.4f}")
    print(f"  {'Purged KFold (contiguous)':<30} {purged_train_mean:>8.4f} {purged_val_mean:>8.4f} {purged_train_mean-purged_val_mean:>8.4f}")
    print(f"  {'Walk-Forward Purged':<30} {wf_train_mean:>8.4f} {wf_val_mean:>8.4f} {wf_train_mean-wf_val_mean:>8.4f}")
    print("=" * 70)

    # Interpretation
    print("\n  Interpretation:")
    if std_val_mean - purged_val_mean > 0.005:
        print(f"  ⚠️  Standard KFold is {(std_val_mean-purged_val_mean)*100:.2f}% higher than Purged.")
        print("      → Temporal leakage detected! Standard KFold inflates the score.")
        print("      → Use Purged KFold for more realistic estimates.")
    elif std_val_mean - purged_val_mean > 0.001:
        print(f"  ⚡ Small difference ({(std_val_mean-purged_val_mean)*100:.2f}%). Minor temporal leakage possible.")
    else:
        print(f"  ✅ Negligible difference ({(std_val_mean-purged_val_mean)*100:.2f}%). No temporal leakage detected.")


if __name__ == "__main__":
    run()
