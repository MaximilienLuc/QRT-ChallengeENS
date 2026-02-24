"""
J4 Hybrid Pipeline — J3 Feature Eng + PCA + Autoencoder + Logistic Regression
===============================================================================
Combines the domain-knowledge features from J3 (rolling stats, momentum,
volatility, interactions) with PCA/Autoencoder dimensionality reduction
and regularized logistic regression.

Pipeline:
  Raw data → J3 Feature Engineering (~80 features)
           → Imputation + Scaling
           → PCA + Autoencoder
           → Regularized Logistic Regression

Usage:
  python run_hybrid_pipeline.py                  # Full data, 30 trials
  python run_hybrid_pipeline.py --sample         # Sample data
  python run_hybrid_pipeline.py --n-trials 5     # Custom Optuna trials
"""

import argparse
import warnings
import sys
import os
import numpy as np
import pandas as pd
import optuna
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))

from pipeline.pca_features import PCAFeatureExtractor
from pipeline.autoencoder import AutoencoderFeatureExtractor
from pipeline.classifier import RegularizedLogistic
from pipeline.purged_kfold import PurgedKFold

# J3 feature engineering — imported via importlib to avoid name collision with J4/pipeline
import importlib.util

_j3_fe_path = os.path.join(os.path.dirname(__file__), '..', 'J3', 'pipeline', 'feature_engineering.py')
_spec = importlib.util.spec_from_file_location('j3_feature_engineering', _j3_fe_path)
_j3_fe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_j3_fe)

RawFeatureGenerator = _j3_fe.RawFeatureGenerator
RollingStatFeatureGenerator = _j3_fe.RollingStatFeatureGenerator
MomentumGenerator = _j3_fe.MomentumGenerator
VolatilityRatioGenerator = _j3_fe.VolatilityRatioGenerator
ShortTermInteractionGenerator = _j3_fe.ShortTermInteractionGenerator
HigherOrderStatsGenerator = _j3_fe.HigherOrderStatsGenerator
SignFlipGenerator = _j3_fe.SignFlipGenerator
VolumeFlowGenerator = _j3_fe.VolumeFlowGenerator
CrossLagCorrelationGenerator = _j3_fe.CrossLagCorrelationGenerator


def j3_feature_generators():
    """Same generators as J3 clean_pipeline + extra generators from J3."""
    return [
        RawFeatureGenerator(
            cols=[f'RET_{i}' for i in range(1, 21)]
               + [f'SIGNED_VOLUME_{i}' for i in range(1, 21)]
               + ['MEDIAN_DAILY_TURNOVER']
        ),
        RollingStatFeatureGenerator(
            cols=[f'RET_{i}' for i in range(1, 21)],
            windows=[5, 10, 20],
            operations=['mean', 'std', 'min', 'max']
        ),
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
        VolatilityRatioGenerator(windows=[(5, 20)]),
        ShortTermInteractionGenerator(max_lag=10),
        HigherOrderStatsGenerator(windows=[5, 10, 20]),
        SignFlipGenerator(windows=[5, 10, 20]),
        VolumeFlowGenerator(windows=[5, 10]),
        CrossLagCorrelationGenerator(windows=[10, 20]),
    ]


def build_j3_features(X, y=None, generators=None, fit=True):
    """Apply J3 feature generators."""
    X_feat = pd.DataFrame(index=X.index)
    for gen in generators:
        if fit and hasattr(gen, 'fit'):
            gen.fit(X, y)
        X_part = gen.transform(X)
        X_feat = pd.concat([X_feat, X_part], axis=1)
    X_feat = X_feat.loc[:, ~X_feat.columns.duplicated()]
    return X_feat


def build_reduced_features(X_scaled, pca_n, ae_latent, ae_lr, ae_dropout, verbose=True):
    """PCA + Autoencoder on already-scaled features."""
    input_dim = X_scaled.shape[1]

    pca = PCAFeatureExtractor(n_components=min(pca_n, input_dim))
    X_pca = pca.fit_transform(X_scaled)

    ae = AutoencoderFeatureExtractor(
        input_dim=input_dim,
        latent_dim=ae_latent,
        lr=ae_lr,
        dropout=ae_dropout,
        n_epochs=100,
        batch_size=512,
        patience=10,
        verbose=False,
    )
    X_ae = ae.fit_transform(X_scaled)

    return np.hstack([X_pca, X_ae]), pca, ae


def transform_reduced(X_scaled, pca, ae):
    """Transform test data with fitted PCA + AE."""
    X_pca = pca.transform(X_scaled)
    X_ae = ae.transform(X_scaled)
    return np.hstack([X_pca, X_ae])


def purged_cv_score(X, y, ts_groups, penalty, C, l1_ratio, n_splits=3):
    """Evaluate with PurgedKFold."""
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=0.02)
    val_scores, train_scores = [], []

    for train_idx, val_idx in pkf.split(X, groups=ts_groups):
        clf = RegularizedLogistic(penalty=penalty, C=C, l1_ratio=l1_ratio)
        y_arr = y.values if hasattr(y, 'values') else y
        clf.fit(X[train_idx], y_arr[train_idx])
        val_scores.append(accuracy_score(y_arr[val_idx], clf.predict(X[val_idx])))
        train_scores.append(accuracy_score(y_arr[train_idx], clf.predict(X[train_idx])))

    return np.mean(val_scores), np.std(val_scores), np.mean(train_scores)


def run(args):
    print("=" * 65)
    print("  J4 HYBRID — J3 FEATURES + PCA + AUTOENCODER + LOGREG")
    print("=" * 65)

    # ── 1. Load ──
    print("\n[1/7] Loading data...")
    suffix = '_sample' if args.sample else ''
    X_train_raw = pd.read_csv(f'../Data/X_train{suffix}.csv')
    y_train_raw = pd.read_csv(f'../Data/y_train{suffix}.csv')
    X_test_raw = pd.read_csv(f'../Data/X_test.csv')

    train_df = X_train_raw.merge(y_train_raw, on='ROW_ID')
    y = (train_df['target'] > 0).astype(int)
    X_train = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_raw.drop(columns=['ROW_ID'])
    test_ids = X_test_raw['ROW_ID']

    ts_groups = X_train['TS'].str.extract(r'(\d+)')[0].astype(int).values

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")

    # ── 2. J3 Feature Engineering ──
    print("\n[2/7] J3 Feature Engineering...")
    generators = j3_feature_generators()
    X_train_feat = build_j3_features(X_train, y, generators, fit=True)
    X_test_feat = build_j3_features(X_test, generators=generators, fit=False)
    print(f"  Generated: {X_train_feat.shape[1]} features")

    # ── 3. Imputation + Scaling ──
    print("\n[3/7] Imputation + Scaling...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train_feat)
    X_test_imp = imputer.transform(X_test_feat)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    input_dim = X_train_scaled.shape[1]
    n_nan = np.isnan(X_train_feat.values).sum()
    print(f"  Imputed {n_nan} NaN, scaled {input_dim} features")

    # ── 4. Optuna ──
    print(f"\n[4/7] Optuna optimization ({args.n_trials} trials)...")

    def objective(trial):
        pca_n = trial.suggest_int('pca_n', 10, min(input_dim, 60))
        ae_latent = trial.suggest_int('ae_latent', 8, 32)
        ae_lr = trial.suggest_float('ae_lr', 1e-4, 1e-2, log=True)
        ae_dropout = trial.suggest_float('ae_dropout', 0.1, 0.5)

        X_combined, _, _ = build_reduced_features(
            X_train_scaled, pca_n, ae_latent, ae_lr, ae_dropout, verbose=False
        )

        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        C = trial.suggest_float('C', 1e-3, 100.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9) if penalty == 'elasticnet' else 0.5

        val_acc, _, train_acc = purged_cv_score(
            X_combined, y, ts_groups, penalty, C, l1_ratio, n_splits=3
        )
        trial.set_user_attr('train_acc', train_acc)
        trial.set_user_attr('gap', train_acc - val_acc)
        return val_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    bp = study.best_params
    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best Purged CV accuracy: {study.best_value:.4f}")
    print(f"  Train-val gap: {study.best_trial.user_attrs['gap']:.4f}")
    print(f"  Best params:")
    for k, v in bp.items():
        print(f"    {k}: {v}")

    # ── 5. Build final features ──
    print("\n[5/7] Building final features with best hyperparams...")
    X_train_final, pca, ae = build_reduced_features(
        X_train_scaled, bp['pca_n'], bp['ae_latent'], bp['ae_lr'], bp['ae_dropout']
    )
    X_test_final = transform_reduced(X_test_scaled, pca, ae)
    print(f"  Feature shape: Train {X_train_final.shape}, Test {X_test_final.shape}")

    # ── 6. Final evaluation (5-fold) ──
    print("\n[6/7] Final 5-fold Purged KFold evaluation...")
    l1_ratio = bp.get('l1_ratio', 0.5)
    val_acc, val_std, train_acc = purged_cv_score(
        X_train_final, y, ts_groups, bp['penalty'], bp['C'], l1_ratio, n_splits=5
    )
    print(f"  Train: {train_acc:.4f} | Val: {val_acc:.4f} (±{val_std:.4f}) | Gap: {train_acc - val_acc:.4f}")

    # ── 7. Train final + submission ──
    print("\n[7/7] Training final model & generating submission...")
    clf = RegularizedLogistic(penalty=bp['penalty'], C=bp['C'], l1_ratio=l1_ratio)
    clf.fit(X_train_final, y)

    preds, threshold = clf.predict_calibrated(X_test_final, y)
    print(f"  Threshold: {threshold:.6f} (target ratio: {y.mean():.4f})")
    print(f"  Predictions: {preds.sum()} pos / {len(preds)} total ({preds.mean()*100:.1f}%)")

    submission = pd.DataFrame({'ROW_ID': test_ids, 'score': preds})
    output_file = 'submission_j4_hybrid_pca_ae_logreg.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file}")

    # Top 5 trials
    print("\n─── Top 5 Trials ───")
    trials_df = study.trials_dataframe().sort_values('value', ascending=False).head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number']):>2}: val={row['value']:.4f}, gap={row['user_attrs_gap']:.4f}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='J4 Hybrid Pipeline')
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    parser.add_argument('--n-trials', type=int, default=30, help='Optuna trials')
    args = parser.parse_args()
    run(args)
