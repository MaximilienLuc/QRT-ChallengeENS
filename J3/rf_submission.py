"""
Random Forest Submission Pipeline — J3
=======================================
Incorporates findings from daily_log_2026_02_17.md:
  1. Tighter adversarial validation threshold (0.60) to catch drifting features
     that hurt LB score (previous 0.70 let 0.618-drift features through).
  2. Mean imputation for NaN handling (RF cannot handle NaNs natively).
  3. Drops GroupedFeatureGenerator — group-level stats computed on training data
     are a major source of train/test distribution mismatch.
  4. Optuna-optimized RF hyperparameters instead of the old J1 defaults.
  5. Lower correlation threshold (0.95) to reduce multicollinearity.
"""

import pandas as pd
import numpy as np
import optuna
import sys
import os
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from pipeline.feature_engineering import (
    RawFeatureGenerator,
    RollingStatFeatureGenerator,
    MomentumGenerator,
    VolatilityRatioGenerator,
    InteractionGenerator,
    ShortTermInteractionGenerator,
)
from pipeline.feature_selection import FeatureEvaluator


def build_features(X, generators, fitted=False, y=None):
    """Apply all feature generators and return a clean DataFrame."""
    X_out = pd.DataFrame(index=X.index)
    for gen in generators:
        if not fitted:
            gen.fit(X, y)
        X_part = gen.transform(X)
        X_out = pd.concat([X_out, X_part], axis=1)
    # Deduplicate columns (ShortTermInteraction overlaps with Interaction)
    X_out = X_out.loc[:, ~X_out.columns.duplicated()]
    return X_out


def run():
    print("=" * 60)
    print("Random Forest Submission Pipeline (J3)")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\nStep 1: Loading Full Data...")
    X_train_raw = pd.read_csv('../Data/X_train.csv')
    y_train_raw = pd.read_csv('../Data/y_train.csv')
    X_test_raw = pd.read_csv('../Data/X_test.csv')

    train_df = X_train_raw.merge(y_train_raw, on='ROW_ID')
    y = (train_df['target'] > 0).astype(int)
    X = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_raw.drop(columns=['ROW_ID'])

    print(f"  Train: {X.shape}  |  Test: {X_test.shape}")

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # NOTE: GroupedFeatureGenerator deliberately excluded — it computes
    #       group-level means on training data that don't generalise to
    #       test, contributing to the CV↔LB gap observed on 2026-02-17.
    # ------------------------------------------------------------------
    print("\nStep 2: Feature Engineering...")
    lag_features = [f'RET_{i}' for i in range(1, 21)]
    volume_features = [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]

    generators = [
        RawFeatureGenerator(cols=lag_features + volume_features + ['MEDIAN_DAILY_TURNOVER']),
        RollingStatFeatureGenerator(
            cols=lag_features,
            windows=[5, 10, 20],
            operations=['mean', 'std', 'min', 'max'],
        ),
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
        VolatilityRatioGenerator(windows=[(5, 20)]),
        InteractionGenerator(),
        ShortTermInteractionGenerator(max_lag=10),
    ]

    X_train_feat = build_features(X, generators, fitted=False, y=y)
    X_test_feat = build_features(X_test, generators, fitted=True)

    print(f"  Features generated: {X_train_feat.shape[1]}")

    # ------------------------------------------------------------------
    # Step 3: Imputation — RF cannot handle NaN natively
    # Mean imputation chosen because missingness is MCAR/MAR (correlation
    # with target < 0.014, per missingness analysis).
    # ------------------------------------------------------------------
    print("\nStep 3: Mean Imputation (RF requires no NaN)...")
    nan_cols = X_train_feat.columns[X_train_feat.isna().any()].tolist()
    print(f"  Columns with NaN: {len(nan_cols)}")

    imputer = SimpleImputer(strategy='mean')
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train_feat),
        columns=X_train_feat.columns,
        index=X_train_feat.index,
    )
    X_test_imp = pd.DataFrame(
        imputer.transform(X_test_feat),
        columns=X_test_feat.columns,
        index=X_test_feat.index,
    )

    # ------------------------------------------------------------------
    # Step 4: Adversarial validation — tighter threshold (0.60)
    # Previous run had AUC 0.618 which slipped under the 0.70 gate,
    # resulting in a 0.5085 LB score. Lowering to 0.60 forces removal
    # of drifting features.
    # ------------------------------------------------------------------
    print("\nStep 4: Adversarial Validation (threshold=0.60)...")
    evaluator = FeatureEvaluator(cv=3)
    drifting = evaluator.check_drift(X_train_imp, X_test_imp, threshold=0.60)
    print(f"  Dropping {len(drifting)} drifting features: {drifting[:10]}")

    X_train_imp = X_train_imp.drop(columns=drifting)
    X_test_imp = X_test_imp.drop(columns=drifting)

    # ------------------------------------------------------------------
    # Step 5: Correlation filter — 0.95 threshold (tighter than previous 0.98)
    # ------------------------------------------------------------------
    print("\nStep 5: Correlation Filter (threshold=0.95)...")
    high_corr = evaluator.check_correlation(X_train_imp, threshold=0.95)
    print(f"  Dropping {len(high_corr)} highly correlated features: {high_corr[:10]}")

    X_final = X_train_imp.drop(columns=high_corr)
    X_test_final = X_test_imp.drop(columns=high_corr)
    print(f"  Final feature count: {X_final.shape[1]}")

    # ------------------------------------------------------------------
    # Step 6: Optuna hyperparameter optimization for Random Forest
    # RF is much slower than LightGBM on large data, so we tune on a
    # stratified subsample (~50K rows) then train the final model on
    # the full dataset.
    # ------------------------------------------------------------------
    print("\nStep 6: Optuna Optimization (Random Forest on subsample)...")

    OPTUNA_SAMPLE_SIZE = 50_000
    if len(X_final) > OPTUNA_SAMPLE_SIZE:
        from sklearn.model_selection import train_test_split
        X_optuna, _, y_optuna, _ = train_test_split(
            X_final, y, train_size=OPTUNA_SAMPLE_SIZE,
            stratify=y, random_state=42,
        )
        print(f"  Subsampled to {X_optuna.shape[0]} rows for tuning")
    else:
        X_optuna, y_optuna = X_final, y

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 200, 600),
            'max_depth': trial.suggest_int('max_depth', 4, 16),
            'min_samples_split': trial.suggest_int('min_samples_split', 5, 50),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 2, 30),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', 0.3, 0.5]),
            'random_state': 42,
            'n_jobs': -1,
        }
        clf = RandomForestClassifier(**params)
        scores = cross_val_score(clf, X_optuna, y_optuna, cv=folds, scoring='accuracy')
        return scores.mean()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=True)

    print(f"\n  Best CV Accuracy (on subsample): {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")

    # ------------------------------------------------------------------
    # Step 7: Train final model & generate submission
    # ------------------------------------------------------------------
    print("\nStep 7: Training Final Model...")
    best_params = study.best_params
    best_params['random_state'] = 42
    best_params['n_jobs'] = -1

    final_rf = RandomForestClassifier(**best_params)
    final_rf.fit(X_final, y)

    preds = final_rf.predict(X_test_final)

    # Submission format must match sample_submission.csv: ROW_ID, prediction
    submission = pd.DataFrame({
        'ROW_ID': X_test_raw['ROW_ID'],
        'prediction': preds,
    })

    output_file = 'submission_rf_optuna.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n  Submission saved to {output_file}")
    print(f"  Predictions: {pd.Series(preds).value_counts().to_dict()}")

    # Log feature importances
    importances = pd.DataFrame({
        'feature': X_final.columns,
        'importance': final_rf.feature_importances_,
    }).sort_values('importance', ascending=False)

    print("\n  Top 15 Feature Importances:")
    print(importances.head(15).to_string(index=False))

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    run()
