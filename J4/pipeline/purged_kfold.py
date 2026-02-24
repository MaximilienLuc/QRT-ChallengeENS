"""
Purged K-Fold Cross-Validation for time-series data.
Prevents temporal leakage by purging samples near fold boundaries.
"""

import numpy as np


class PurgedKFold:
    def __init__(self, n_splits=5, purge_pct=0.02):
        self.n_splits = n_splits
        self.purge_pct = purge_pct

    def split(self, X, y=None, groups=None):
        """
        Generate train/val indices with temporal purging.

        Args:
            X: feature matrix (used only for indexing)
            y: ignored
            groups: array of integer timestamps per sample
        """
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
