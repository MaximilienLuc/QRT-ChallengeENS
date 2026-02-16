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
