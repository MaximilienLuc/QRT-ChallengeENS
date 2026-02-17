import pandas as pd
import numpy as np
import sys
import os
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer

# Add pipeline directory to path if not already there
sys.path.append(os.path.abspath('.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator,
    RollingStatFeatureGenerator,
    GroupedFeatureGenerator,
    MomentumGenerator,
    VolatilityRatioGenerator,
    InteractionGenerator,
    ShortTermInteractionGenerator
)

def run():
    print("Step 1: Loading Sample Data...")
    try:
        X_train = pd.read_csv('Data/X_train_sample.csv')
        y_train = pd.read_csv('Data/y_train_sample.csv')
    except FileNotFoundError:
        print("Error: Sample data not found. Please ensure Data/X_train_sample.csv exists.")
        return

    # Merge target for evaluation convenience
    df = X_train.merge(y_train, on='ROW_ID')
    y = (df['target'] > 0).astype(int) # Binary target
    X = df.drop(columns=['target', 'ROW_ID'])
    print(f"Data loaded: {X.shape}")

    print("Step 2: Defining Pipeline & Generating Features (With NaNs)...")
    # Base features
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

    X_transformed = pd.DataFrame(index=X.index)

    # Apply all generators
    for gen in generators:
        gen.fit(X, y)
        X_part = gen.transform(X)
        X_transformed = pd.concat([X_transformed, X_part], axis=1)

    # Remove duplicate columns
    X_transformed = X_transformed.loc[:, ~X_transformed.columns.duplicated()]

    print(f"Generated Features: {X_transformed.shape[1]}")
    
    # Check for NaNs
    nan_count = X_transformed.isna().sum().sum()
    print(f"Total NaNs in dataset: {nan_count}")
    print(f"Columns with NaNs: {X_transformed.columns[X_transformed.isna().any()].tolist()}")

    print("\nStep 3: Comparing Strategies...")
    
    strategies = {
        'Baseline (Fill 0)': lambda df: df.fillna(0),
        'Native LightGBM (Leave NaN)': lambda df: df,  # LightGBM handles NaNs natively
        'Mean Imputation': lambda df: pd.DataFrame(SimpleImputer(strategy='mean').fit_transform(df), columns=df.columns, index=df.index)
    }

    results = {}
    
    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    # Basic LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_jobs': -1,
        'random_state': 42
    }

    for name, strategy_fn in strategies.items():
        print(f"\nEvaluating: {name}...")
        
        # Apply strategy
        X_strat = strategy_fn(X_transformed.copy())
        
        # CV
        clf = lgb.LGBMClassifier(**params)
        cv_scores = cross_val_score(clf, X_strat, y, cv=folds, scoring='accuracy')
        
        mean_score = cv_scores.mean()
        std_score = cv_scores.std()
        results[name] = (mean_score, std_score)
        print(f"  -> Accuracy: {mean_score:.4f} (+/- {std_score:.4f})")

    print("\n=== Summary of Results ===")
    print(f"{'Strategy':<30} | {'Accuracy':<10} | {'Std Dev':<10}")
    print("-" * 56)
    for name, (mean, std) in results.items():
        print(f"{name:<30} | {mean:.4f}     | {std:.4f}")

if __name__ == "__main__":
    run()
