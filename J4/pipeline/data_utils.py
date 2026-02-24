"""
Data loading, imputation, and scaling utilities for J4 pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def load_data(data_dir='../Data', sample=False):
    """
    Load train/test data, merge target, return clean arrays.

    Returns:
        X_train: DataFrame of numeric features (no ROW_ID, no TS)
        X_test:  DataFrame of numeric features (no ROW_ID)
        y:       Series, binary target (return > 0)
        ts_groups: ndarray, integer timestamps for PurgedKFold
        test_ids:  Series, ROW_IDs for submission
    """
    suffix = '_sample' if sample else ''
    X_train_raw = pd.read_csv(f'{data_dir}/X_train{suffix}.csv')
    y_train_raw = pd.read_csv(f'{data_dir}/y_train{suffix}.csv')
    X_test_raw = pd.read_csv(f'{data_dir}/X_test.csv')

    # Merge and extract target
    train_df = X_train_raw.merge(y_train_raw, on='ROW_ID')
    y = (train_df['target'] > 0).astype(int)

    # Separate metadata from features
    X_train = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_raw.drop(columns=['ROW_ID'])
    test_ids = X_test_raw['ROW_ID']

    # Extract timestamp groups for PurgedKFold
    ts_groups = X_train['TS'].str.extract(r'(\d+)')[0].astype(int).values

    # Drop TS (not a numeric feature)
    X_train = X_train.drop(columns=['TS'])
    if 'TS' in X_test.columns:
        X_test = X_test.drop(columns=['TS'])

    # Drop non-numeric columns (e.g., ALLOCATION is categorical string)
    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  Dropping non-numeric columns: {non_numeric}")
        X_train = X_train.drop(columns=non_numeric)
        X_test = X_test.drop(columns=[c for c in non_numeric if c in X_test.columns])

    print(f"  Loaded: Train {X_train.shape}, Test {X_test.shape}")
    print(f"  Features: {list(X_train.columns[:5])}... ({X_train.shape[1]} total)")
    print(f"  Target balance: {y.mean():.3f} positive")

    return X_train, X_test, y, ts_groups, test_ids


def preprocess(X_train, X_test):
    """
    Impute NaNs (mean) and scale features (StandardScaler).
    Both fitted on train only.

    Returns:
        X_train_scaled: ndarray
        X_test_scaled:  ndarray
        imputer:        fitted SimpleImputer
        scaler:         fitted StandardScaler
        feature_names:  list of column names
    """
    feature_names = list(X_train.columns)

    # Imputation
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # Scaling (critical for PCA and LogReg)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    n_nan_train = np.isnan(X_train.values).sum()
    n_nan_test = np.isnan(X_test.values).sum()
    print(f"  Imputed: {n_nan_train} NaN (train), {n_nan_test} NaN (test)")
    print(f"  Scaled: mean≈0, std≈1")

    return X_train_scaled, X_test_scaled, imputer, scaler, feature_names
