import pandas as pd
import numpy as np

class FeatureGenerator:
    """Base class for feature generation strategies."""
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X

class RawFeatureGenerator(FeatureGenerator):
    """Pass-through generator that returns specified features."""
    def __init__(self, cols):
        self.cols = cols

    def transform(self, X):
        return X[self.cols].copy()

class RollingStatFeatureGenerator(FeatureGenerator):
    """Generates rolling statistics (mean, std, etc.) on existing features (e.g., returns)."""
    def __init__(self, cols, windows, operations=['mean', 'std']):
        self.cols = cols 
        self.windows = windows
        self.operations = operations

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        
        for w in self.windows:
            relevant_cols = [f'RET_{i}' for i in range(1, w + 1)]
            available_cols = [c for c in relevant_cols if c in X.columns]
            if len(available_cols) < w:
                continue
                
            if 'mean' in self.operations:
                X_new[f'RET_MEAN_{w}'] = X[available_cols].mean(axis=1)
            if 'std' in self.operations:
                X_new[f'RET_STD_{w}'] = X[available_cols].std(axis=1)
            if 'min' in self.operations:
                X_new[f'RET_MIN_{w}'] = X[available_cols].min(axis=1)
            if 'max' in self.operations:
                X_new[f'RET_MAX_{w}'] = X[available_cols].max(axis=1)
            if 'skew' in self.operations:
                X_new[f'RET_SKEW_{w}'] = X[available_cols].skew(axis=1)
            if 'kurt' in self.operations:
                X_new[f'RET_KURT_{w}'] = X[available_cols].kurt(axis=1)

        return X_new

class MomentumGenerator(FeatureGenerator):
    """Generates momentum features (short vs long term trends)."""
    def __init__(self, windows=[(1, 5), (1, 20), (5, 20)]):
        """
        windows: list of tuples (short, long) to compare.
        """
        self.windows = windows

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        
        for short, long in self.windows:
            col_short = f'RET_{short}' if short <= 20 else None
            col_long = f'RET_{long}' if long <= 20 else None
            
            # 1. Point-to-Point Momentum (RET_1 - RET_5)
            if col_short and col_long and col_short in X.columns and col_long in X.columns:
                 X_new[f'MOM_POINT_{short}_{long}'] = X[col_short] - X[col_long]

            # 2. MA Crossover Proxy
            cols_short = [f'RET_{i}' for i in range(1, short + 1)]
            cols_long = [f'RET_{i}' for i in range(1, long + 1)]
            
            valid_short = [c for c in cols_short if c in X.columns]
            valid_long = [c for c in cols_long if c in X.columns]
            
            if len(valid_short) == short and len(valid_long) == long:
                 mean_short = X[valid_short].mean(axis=1)
                 mean_long = X[valid_long].mean(axis=1)
                 X_new[f'MOM_MA_{short}_{long}'] = mean_short - mean_long
        
        return X_new

class VolatilityRatioGenerator(FeatureGenerator):
    """Generates volatility ratios (short term vol / long term vol)."""
    def __init__(self, windows=[(5, 20)]):
        self.windows = windows

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        
        for short, long in self.windows:
            cols_short = [f'RET_{i}' for i in range(1, short + 1)]
            cols_long = [f'RET_{i}' for i in range(1, long + 1)]
            
            valid_short = [c for c in cols_short if c in X.columns]
            valid_long = [c for c in cols_long if c in X.columns]
            
            if len(valid_short) == short and len(valid_long) == long:
                 std_short = X[valid_short].std(axis=1)
                 std_long = X[valid_long].std(axis=1)
                 
                 X_new[f'VOL_RATIO_{short}_{long}'] = std_short / (std_long + 1e-9)
                 
                 if 'RET_1' in X.columns:
                     X_new[f'SHARPE_PROXY_{long}'] = X['RET_1'] / (std_long + 1e-9)

        return X_new

class GroupedFeatureGenerator(FeatureGenerator):
    """Generates features aggregated by group (e.g., SECTOR, ALLOCATION)."""
    def __init__(self, group_col, target_cols, operations=['mean']):
        self.group_col = group_col
        self.target_cols = target_cols
        self.operations = operations
        self.stats = {}

    def fit(self, X, y=None):
        for op in self.operations:
            valid_targets = [c for c in self.target_cols if c in X.columns]
            if not valid_targets:
                continue
            self.stats[op] = X.groupby(self.group_col)[valid_targets].agg(op)
        return self

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for op, stats_df in self.stats.items():
            for col in self.target_cols:
                if col not in X.columns:
                    continue
                
                feature_name = f'{self.group_col}_{col}_{op}'
                mapped_values = X[self.group_col].map(stats_df[col])
                X_new[feature_name] = mapped_values
                
                if op == 'mean':
                    X_new[f'{col}_rel_{self.group_col}'] = X[col] - mapped_values
                
        return X_new

class InteractionGenerator(FeatureGenerator):
    """Generates interaction features (e.g. Return * Volume)."""
    def __init__(self):
        pass

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        
        # 1. Price * Volume (Force/Impact)
        # For the most recent lags
        for i in range(1, 6): # Last 5 days
            ret_col = f'RET_{i}'
            vol_col = f'SIGNED_VOLUME_{i}'
            
            if ret_col in X.columns and vol_col in X.columns:
                X_new[f'RET_x_VOL_{i}'] = X[ret_col] * X[vol_col]
        
        # 2. Return * Turnover (Dollar Impact)
        if 'MEDIAN_DAILY_TURNOVER' in X.columns:
            for i in range(1, 6):
                if f'RET_{i}' in X.columns:
                    X_new[f'RET_{i}_x_TURNOVER'] = X[f'RET_{i}'] * X['MEDIAN_DAILY_TURNOVER']

        return X_new

class ShortTermInteractionGenerator(FeatureGenerator):
    """Generates short-term interactions (< 10 lags) between Return and Volume."""
    def __init__(self, max_lag=10):
        self.max_lag = max_lag

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        
        # 1. Synchronous Volume-Price (RET_i * SIGNED_VOLUME_i for i <= max_lag)
        # 2. Cumulative "Money Flow" (Sum of RET * VOL over 5 days)
        
        cumulative_flow = 0
        
        for i in range(1, self.max_lag + 1):
            ret_col = f'RET_{i}'
            vol_col = f'SIGNED_VOLUME_{i}'
            
            if ret_col in X.columns and vol_col in X.columns:
                # Synchronous
                term = X[ret_col] * X[vol_col]
                X_new[f'RET_x_VOL_{i}'] = term
                
                # Cross-Sectional Ranking Proxy? Maybe later.
                
                if i <= 5:
                    cumulative_flow += term
        
        X_new['CUMUL_FLOW_5'] = cumulative_flow

        # 3. Volume-Weighted Return (RET_i * (VOL_i / Mean_VOL_Asset))
        # Requires asset-specific mean volume, but here we can try: 
        # RET_i * (VOL_i / TURNOVER_MEDIAN?)
        if 'MEDIAN_DAILY_TURNOVER' in X.columns:
             for i in range(1, 6):
                ret_col = f'RET_{i}'
                vol_col = f'SIGNED_VOLUME_{i}'
                if ret_col in X.columns and vol_col in X.columns:
                    # Normalized Impact
                    X_new[f'RET_VOL_NORM_{i}'] = (X[ret_col] * X[vol_col]) / (X['MEDIAN_DAILY_TURNOVER'] + 1e-9)

        return X_new


class HigherOrderStatsGenerator(FeatureGenerator):
    """Skewness and Kurtosis of returns over rolling windows."""
    def __init__(self, windows=[5, 10, 20]):
        self.windows = windows

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for w in self.windows:
            cols = [f'RET_{i}' for i in range(1, w + 1)]
            avail = [c for c in cols if c in X.columns]
            if len(avail) < w:
                continue
            vals = X[avail]
            X_new[f'RET_SKEW_{w}'] = vals.skew(axis=1)
            X_new[f'RET_KURT_{w}'] = vals.kurt(axis=1)
        return X_new


class SignFlipGenerator(FeatureGenerator):
    """Features based on sign patterns in returns: reversals, streaks, directional bias."""
    def __init__(self, windows=[5, 10, 20]):
        self.windows = windows

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for w in self.windows:
            cols = [f'RET_{i}' for i in range(1, w + 1)]
            avail = [c for c in cols if c in X.columns]
            if len(avail) < w:
                continue
            signs = np.sign(X[avail].values)  # -1, 0, +1

            # 1. Count sign flips (high = choppy market)
            flips = np.sum(np.diff(signs, axis=1) != 0, axis=1)
            X_new[f'SIGN_FLIPS_{w}'] = flips

            # 2. Ratio of positive returns
            X_new[f'POS_RATIO_{w}'] = np.mean(signs > 0, axis=1)

            # 3. Max consecutive same-sign streak (from most recent)
            streak = np.ones(len(X))
            for i in range(1, signs.shape[1]):
                streak += (signs[:, i] == signs[:, 0]).astype(float) * (streak == i).astype(float)
            X_new[f'STREAK_{w}'] = streak

        return X_new


class VolumeFlowGenerator(FeatureGenerator):
    """Order flow imbalance, VWAP proxy, volume acceleration."""
    def __init__(self, windows=[5, 10]):
        self.windows = windows

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for w in self.windows:
            vol_cols = [f'SIGNED_VOLUME_{i}' for i in range(1, w + 1)]
            ret_cols = [f'RET_{i}' for i in range(1, w + 1)]
            avail_vol = [c for c in vol_cols if c in X.columns]
            avail_ret = [c for c in ret_cols if c in X.columns]

            if len(avail_vol) >= 2:
                vols = X[avail_vol]
                # 1. Net volume imbalance
                X_new[f'VOL_IMBALANCE_{w}'] = vols.sum(axis=1)
                # 2. Volume std (activity dispersion)
                X_new[f'VOL_STD_{w}'] = vols.std(axis=1)
                # 3. Volume trend (recent - old half)
                half = len(avail_vol) // 2
                recent = vols.iloc[:, :half].sum(axis=1)
                older = vols.iloc[:, half:].sum(axis=1)
                X_new[f'VOL_ACCEL_{w}'] = recent - older

            if len(avail_ret) == w and len(avail_vol) == w:
                # 4. Volume-weighted average return (VWAP proxy)
                ret_vals = X[avail_ret].values
                vol_vals = np.abs(X[avail_vol].values) + 1e-9
                X_new[f'VWAP_PROXY_{w}'] = np.sum(ret_vals * vol_vals, axis=1) / np.sum(vol_vals, axis=1)

        # 5. Abs volume / turnover ratio (normalized activity)
        if 'MEDIAN_DAILY_TURNOVER' in X.columns and 'SIGNED_VOLUME_1' in X.columns:
            X_new['VOL_TURNOVER_RATIO'] = np.abs(X['SIGNED_VOLUME_1']) / (X['MEDIAN_DAILY_TURNOVER'] + 1e-9)

        return X_new


class CrossLagCorrelationGenerator(FeatureGenerator):
    """Cross-correlation between RET and VOL, and auto-correlation of RET."""
    def __init__(self, windows=[10, 20]):
        self.windows = windows

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        for w in self.windows:
            ret_cols = [f'RET_{i}' for i in range(1, w + 1)]
            vol_cols = [f'SIGNED_VOLUME_{i}' for i in range(1, w + 1)]
            avail_ret = [c for c in ret_cols if c in X.columns]
            avail_vol = [c for c in vol_cols if c in X.columns]

            if len(avail_ret) == w and len(avail_vol) == w:
                # Row-wise correlation between RET and VOL vectors
                ret_vals = X[avail_ret].values
                vol_vals = X[avail_vol].values
                # Demean
                ret_dm = ret_vals - ret_vals.mean(axis=1, keepdims=True)
                vol_dm = vol_vals - vol_vals.mean(axis=1, keepdims=True)
                num = np.sum(ret_dm * vol_dm, axis=1)
                den = np.sqrt(np.sum(ret_dm**2, axis=1) * np.sum(vol_dm**2, axis=1)) + 1e-9
                X_new[f'CORR_RET_VOL_{w}'] = num / den

            if len(avail_ret) == w and w >= 6:
                # Auto-correlation of RET: corr(RET[1:w/2], RET[w/2+1:w])
                half = w // 2
                recent = X[avail_ret[:half]].values
                older = X[avail_ret[half:2*half]].values
                r_dm = recent - recent.mean(axis=1, keepdims=True)
                o_dm = older - older.mean(axis=1, keepdims=True)
                num = np.sum(r_dm * o_dm, axis=1)
                den = np.sqrt(np.sum(r_dm**2, axis=1) * np.sum(o_dm**2, axis=1)) + 1e-9
                X_new[f'AUTOCORR_RET_{w}'] = num / den

        return X_new


class AllocationFeatureGenerator(FeatureGenerator):
    """Grouped stats by ALLOCATION (sector-like grouping, currently unused)."""
    def __init__(self, target_cols=['RET_1', 'SIGNED_VOLUME_1'], operations=['mean']):
        self.target_cols = target_cols
        self.operations = operations
        self.stats = {}

    def fit(self, X, y=None):
        if 'ALLOCATION' not in X.columns:
            return self
        for op in self.operations:
            valid = [c for c in self.target_cols if c in X.columns]
            if valid:
                self.stats[op] = X.groupby('ALLOCATION')[valid].agg(op)
        return self

    def transform(self, X):
        X_new = pd.DataFrame(index=X.index)
        if 'ALLOCATION' not in X.columns or not self.stats:
            return X_new
        for op, stats_df in self.stats.items():
            for col in self.target_cols:
                if col not in X.columns or col not in stats_df.columns:
                    continue
                mapped = X['ALLOCATION'].map(stats_df[col])
                X_new[f'ALLOC_{col}_{op}'] = mapped
                if op == 'mean':
                    X_new[f'{col}_rel_ALLOC'] = X[col] - mapped
        return X_new
