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

    X_transformed = X_transformed.fillna(0)
    print(f"Generated Features: {X_transformed.shape[1]}")

    print("Step 4: Selection & Evaluation...")
    evaluator = FeatureEvaluator(cv=3)

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
