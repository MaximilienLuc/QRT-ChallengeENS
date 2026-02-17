import pandas as pd
import numpy as np
import mlflow
import sys
import os
import json

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
from pipeline.feature_selection import (
    FeatureEvaluator,
    plot_feature_importance,
    log_experiment_to_mlflow
)

def run():
    # Set MLflow experiment
    # Ensure the experiment exists or create it
    try:
        mlflow.create_experiment("QRT_Feature_Engineering")
    except:
        mlflow.set_experiment("QRT_Feature_Engineering")

    print("Step 1: Loading Data...")
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

    print("Step 2: Defining Pipeline...")
    # Base features
    lag_features = [f'RET_{i}' for i in range(1, 21)]
    volume_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

    generators = [
        # 1. Raw Lags
        RawFeatureGenerator(cols=lag_features + volume_features + ['MEDIAN_DAILY_TURNOVER']),
        
        # 2. Rolling Stats on Lags
        RollingStatFeatureGenerator(cols=lag_features, windows=[5, 10, 20], operations=['mean', 'std', 'min', 'max']),

        # 3. Momentum Features (Trend)
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
    
        # 4. Volatility Comparisons (Regime changes)
        VolatilityRatioGenerator(windows=[(5, 20)]),
    
        # 5. Grouped Stats (Sector Relative Performance)
        GroupedFeatureGenerator(group_col='GROUP', target_cols=['RET_1', 'SIGNED_VOLUME_1'], operations=['mean']),

        # 6. Interactions (Standard)
        # InteractionGenerator(), # Disabling general interactions to focus on short term as requested? Or keep both?
        # Let's keep both to compare.
        InteractionGenerator(),

        # 7. Short-Term Interactions (< 10 lags, Volume weighted)
        ShortTermInteractionGenerator(max_lag=10)
    ]

    print("Step 3: Generating Features...")
    X_transformed = pd.DataFrame(index=X.index)

    # Apply all generators
    for gen in generators:
        gen.fit(X, y)
        X_part = gen.transform(X)
        X_transformed = pd.concat([X_transformed, X_part], axis=1)

    # X_transformed = X_transformed.fillna(0) # Removed to let LightGBM handle NaNs natively
    print(f"Generated Features: {X_transformed.shape[1]}")
    
    # Remove duplicate columns
    X_transformed = X_transformed.loc[:, ~X_transformed.columns.duplicated()]

    print("Step 4: Selection & Evaluation...")
    evaluator = FeatureEvaluator(cv=3)

    # 0. Check Drift
    print("Checking for Drifting Features...")
    # For sample run, we split X for drift check simulation or use X_test if available (but X_test is not loaded here usually? Ah, we use X_train sample)
    # The sample run only loads X_train_sample. It doesn't load X_test.
    # So we can't really do drift check here unless we load X_test.
    # Let's skip drift check for run_pipeline.py based on sample data unless we load X_test full.
    # Actually prompt says "Verify with sample data". Better to load X_test header/sample if possible.
    # Let's mock it or just check correlation.
    
    # NOTE: To properly verify drift check, we need Test data.
    # Let's try to load X_test.csv if it exists (it does).
    try:
        X_test_real = pd.read_csv('Data/X_test.csv')
        # Generate features for X_test_real (subset or full? Full might be slow)
        # Let's take a sample of X_test equal to X_train size for quick check
        X_test_sample = X_test_real.sample(n=min(len(X), 10000), random_state=42).drop(columns=['ROW_ID'])
        
        # Reset index to ensure alignment and uniqueness
        X_test_sample = X_test_sample.reset_index(drop=True)
        
        print("Generating features for Test Sample for Drift Check...")
        X_test_transformed_sample = pd.DataFrame(index=X_test_sample.index)
        for gen in generators:
            # Debugging
            # print(f"Gen: {gen}")
            X_part = gen.transform(X_test_sample)
            X_part = X_part.reset_index(drop=True)
            
            # Ensure target has unique index
            X_test_transformed_sample = X_test_transformed_sample.reset_index(drop=True)
            
            # print(f"Part shape: {X_part.shape}, Transformed shape: {X_test_transformed_sample.shape}")
            X_test_transformed_sample = pd.concat([X_test_transformed_sample, X_part], axis=1)
        
        # Remove duplicates
        X_test_transformed_sample = X_test_transformed_sample.loc[:, ~X_test_transformed_sample.columns.duplicated()]

        drifting_features = evaluator.check_drift(X_transformed, X_test_transformed_sample, threshold=0.60) # Lower threshold for sample
        print(f"Dropping {len(drifting_features)} drifting features.")
        X_transformed = X_transformed.drop(columns=drifting_features)

    except Exception as e:
        print(f"Could not run drift check on sample: {e}")

    # 1. Check Correlation
    high_corr_features = evaluator.check_correlation(X_transformed, threshold=0.95)
    print(f"Dropping {len(high_corr_features)} highly correlated features.")
    X_selected = X_transformed.drop(columns=high_corr_features)

    # 2. Evaluate with CV
    print("Starting MLflow run...")
    with mlflow.start_run(run_name="Short_Term_Interactions"):
        mean_score, std_score, importances = evaluator.evaluate_features(X_selected, y)
        
        print(f"CV Score: {mean_score:.4f} (+/- {std_score:.4f})")
        
        # Log to MLflow
        mlflow.log_param("n_features_generated", X_transformed.shape[1])
        mlflow.log_param("n_features_selected", X_selected.shape[1])
        mlflow.log_metric("cv_accuracy_mean", mean_score)
        mlflow.log_metric("cv_accuracy_std", std_score)
        
        # Save Importance Plot
        plot_feature_importance(importances, save_path="feature_importance.png")
        mlflow.log_artifact("feature_importance.png")

        # Save list of selected features
        with open("selected_features.json", "w") as f:
            json.dump(X_selected.columns.tolist(), f)
        mlflow.log_artifact("selected_features.json")
    
    print("Pipeline run completed successfully.")

if __name__ == "__main__":
    run()
