import pandas as pd
import numpy as np
import sys
import os

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

    print("Step 2: Generating Features (to introduce NaNs)...")
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

    # Remove duplicates
    X_transformed = X_transformed.loc[:, ~X_transformed.columns.duplicated()]

    print(f"Generated Features: {X_transformed.shape[1]}")
    
    print("\nStep 3: Analyzing Missingness...")
    
    # Calculate missing % per column
    missing_percent = X_transformed.isnull().mean() * 100
    missing_cols = missing_percent[missing_percent > 0].sort_values(ascending=False)
    
    print("\nTop features with missing values (%):")
    print(missing_cols.head(10))

    print("\nStep 4: Analyzing Correlation with Target (MNAR Check)...")
    # Create a binary matrix: 1 if missing, 0 if present
    missing_indicator = X_transformed[missing_cols.index].isnull().astype(int)

    # Calculate correlation with Target
    correlations = {}
    for col in missing_indicator.columns:
        if missing_indicator[col].nunique() > 1: # Only if there is variance
            corr = np.corrcoef(missing_indicator[col], y)[0, 1]
            correlations[col] = corr
    
    if not correlations:
        print("No missing values found or constant missingness.")
        return

    corr_series = pd.Series(correlations).sort_values(key=abs, ascending=False)
    
    print("\nTop correlations between Missingness and Target:")
    print(corr_series.head(10))
    
    # simple heuristic
    max_corr = abs(corr_series.iloc[0]) if not corr_series.empty else 0
    print(f"\nMax Correlation: {max_corr:.4f}")
    
    if max_corr > 0.05:
        print("Conclusion: Likely MNAR (Missing Not At Random) or Informative Missingness.")
        print("Action: Preserve NaNs or use Binary Indicator features.")
    else:
        print("Conclusion: Likely MAR/MCAR (Low correlation with target).")
        print("Action: Imputation might be safe, but Native Handling still often best for Tree models.")

if __name__ == "__main__":
    run()
