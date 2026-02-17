import pandas as pd
import numpy as np
import mlflow
import sys
import os
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score

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
from pipeline.feature_selection import FeatureEvaluator

def run():
    mlflow.set_experiment("QRT_Full_Submission_Optuna")

    print("Step 1: Loading Full Data...")
    try:
        X_train_full = pd.read_csv('Data/X_train.csv')
        y_train_full = pd.read_csv('Data/y_train.csv')
        X_test_full = pd.read_csv('Data/X_test.csv')
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Preprocess Y
    train_df = X_train_full.merge(y_train_full, on='ROW_ID')
    y = (train_df['target'] > 0).astype(int) # Binary target
    X = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_full.drop(columns=['ROW_ID'])

    print(f"Training Data: {X.shape}")
    print(f"Test Data: {X_test.shape}")

    print("Step 2: Feature Engineering...")
    lag_features = [f'RET_{i}' for i in range(1, 21)]
    volume_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

    generators = [
        RawFeatureGenerator(cols=lag_features + volume_features + ['MEDIAN_DAILY_TURNOVER']),
        RollingStatFeatureGenerator(cols=lag_features, windows=[5, 10, 20], operations=['mean', 'std', 'min', 'max']),
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
        VolatilityRatioGenerator(windows=[(5, 20)]),
        GroupedFeatureGenerator(group_col='GROUP', target_cols=['RET_1', 'SIGNED_VOLUME_1'], operations=['mean']),
        ShortTermInteractionGenerator(max_lag=10)
    ]

    print("Generating Features for Training Set...")
    X_transformed = pd.DataFrame(index=X.index)
    for gen in generators:
        gen.fit(X, y)
        X_part = gen.transform(X)
        X_transformed = pd.concat([X_transformed, X_part], axis=1)
    # X_transformed = X_transformed.fillna(0) # Removed to let LightGBM handle NaNs natively

    print("Generating Features for Test Set...")
    X_test_transformed = pd.DataFrame(index=X_test.index)
    for gen in generators:
        X_part = gen.transform(X_test)
        X_test_transformed = pd.concat([X_test_transformed, X_part], axis=1)
    # X_test_transformed = X_test_transformed.fillna(0) # Removed to let LightGBM handle NaNs natively

    print(f"Generated Features: {X_transformed.shape[1]}")
    
    # Remove duplicate columns
    X_transformed = X_transformed.loc[:, ~X_transformed.columns.duplicated()]
    X_test_transformed = X_test_transformed.loc[:, ~X_test_transformed.columns.duplicated()]

    print("Step 3: Feature Selection...")
    evaluator = FeatureEvaluator(cv=3)
    print("Checking for Drifting Features (Adversarial Validation)...")
    drifting_features = evaluator.check_drift(X_transformed, X_test_transformed, threshold=0.70)
    print(f"Dropping {len(drifting_features)} drifting features.")
    
    # Drop drifting features first
    X_transformed = X_transformed.drop(columns=drifting_features)
    X_test_transformed = X_test_transformed.drop(columns=drifting_features)

    high_corr_features = evaluator.check_correlation(X_transformed, threshold=0.98)
    print(f"Dropping {len(high_corr_features)} highly correlated features.")
    
    X_final = X_transformed.drop(columns=high_corr_features)
    X_test_final = X_test_transformed.drop(columns=high_corr_features)
    print(f"Final Feature Count: {X_final.shape[1]}")

    print("Step 4: Optuna Optimization...")
    def objective(trial):
        param = {
            'objective': 'binary',
            'metric': 'binary_error',
            'verbosity': -1,
            'boosting_type': 'gbdt',
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 20, 150),
            'max_depth': trial.suggest_int('max_depth', -1, 12),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        }
        folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        clf = lgb.LGBMClassifier(**param, random_state=42, n_jobs=-1)
        cv_scores = cross_val_score(clf, X_final, y, cv=folds, scoring='accuracy')
        return cv_scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20) # 20 trials for reasonable runtime

    print("Best params found:")
    print(study.best_params)

    print("Step 5: Retraining and Submission...")
    best_params = study.best_params
    best_params.update({
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': -1,
        'boosting_type': 'gbdt'
    })

    final_model = lgb.LGBMClassifier(**best_params, random_state=42, n_jobs=-1)
    final_model.fit(X_final, y)

    # Predict probabilities first
    probs = final_model.predict_proba(X_test_final)[:, 1]
    
    # Convert to binary predictions (Threshold 0.5)
    preds = (probs > 0.5).astype(int)

    submission = pd.DataFrame()
    submission['ROW_ID'] = X_test_full['ROW_ID']
    submission['score'] = preds # Binary target as requested

    output_file = 'submission_lgbm_optuna_binary.csv'
    submission.to_csv(output_file, index=False)
    print(f"Submission saved to {output_file}")

if __name__ == "__main__":
    run()
