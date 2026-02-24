"""
J4 Pure Pipeline — PCA + Autoencoder + Logistic Regression
==========================================================
Radically different from J1–J3:
  - No hand-crafted features (rolling stats, momentum, etc.)
  - Raw features → PCA + Autoencoder → Regularized Logistic Regression
  - Optuna optimizes: PCA n_components, AE latent_dim/lr/dropout, LogReg C/penalty

Usage:
  python run_pipeline.py                  # Full data, 30 trials
  python run_pipeline.py --sample         # Sample data for quick test
  python run_pipeline.py --n-trials 5     # Custom number of Optuna trials
"""

import argparse
import warnings
import sys
import os
import numpy as np
import pandas as pd
import optuna
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(__file__))

from pipeline.data_utils import load_data, preprocess
from pipeline.pca_features import PCAFeatureExtractor
from pipeline.autoencoder import AutoencoderFeatureExtractor
from pipeline.classifier import RegularizedLogistic
from pipeline.purged_kfold import PurgedKFold


def build_features(X_train_scaled, X_test_scaled, pca_n, ae_latent, ae_lr, ae_dropout):
    """
    Build feature matrix from PCA + Autoencoder.
    Returns concatenated features for train and test.
    """
    input_dim = X_train_scaled.shape[1]

    # PCA
    pca = PCAFeatureExtractor(n_components=pca_n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Autoencoder
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
    X_train_ae = ae.fit_transform(X_train_scaled)
    X_test_ae = ae.transform(X_test_scaled)

    # Concatenate
    X_train_combined = np.hstack([X_train_pca, X_train_ae])
    X_test_combined = np.hstack([X_test_pca, X_test_ae])

    return X_train_combined, X_test_combined


def purged_cv_score(X_combined, y, ts_groups, penalty, C, l1_ratio, n_splits=3):
    """
    Evaluate logistic regression with PurgedKFold.
    Returns: val_acc_mean, val_acc_std, train_acc_mean
    """
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=0.02)
    val_scores, train_scores = [], []

    for train_idx, val_idx in pkf.split(X_combined, groups=ts_groups):
        clf = RegularizedLogistic(penalty=penalty, C=C, l1_ratio=l1_ratio)
        clf.fit(X_combined[train_idx], y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx])

        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        val_scores.append(accuracy_score(y_val_fold, clf.predict(X_combined[val_idx])))
        train_scores.append(accuracy_score(y_train_fold, clf.predict(X_combined[train_idx])))

    return np.mean(val_scores), np.std(val_scores), np.mean(train_scores)


def run(args):
    print("=" * 65)
    print("  J4 — PURE PIPELINE: PCA + AUTOENCODER + LOGISTIC REGRESSION")
    print("=" * 65)

    # ── 1. Load ──
    print("\n[1/6] Loading data...")
    X_train, X_test, y, ts_groups, test_ids = load_data(
        data_dir='../Data', sample=args.sample
    )

    # ── 2. Preprocess ──
    print("\n[2/6] Preprocessing (imputation + scaling)...")
    X_train_scaled, X_test_scaled, imputer, scaler, feature_names = preprocess(X_train, X_test)
    input_dim = X_train_scaled.shape[1]
    print(f"  Input dimension: {input_dim}")

    # ── 3. Optuna ──
    print(f"\n[3/6] Optuna optimization ({args.n_trials} trials, 3-fold Purged KFold)...")

    def objective(trial):
        # PCA hyperparameters
        pca_n = trial.suggest_int('pca_n_components', 5, min(input_dim, 35))

        # Autoencoder hyperparameters
        ae_latent = trial.suggest_int('ae_latent_dim', 8, 32)
        ae_lr = trial.suggest_float('ae_lr', 1e-4, 1e-2, log=True)
        ae_dropout = trial.suggest_float('ae_dropout', 0.1, 0.5)

        # Build features with these hyperparams
        X_train_feat, _ = build_features(
            X_train_scaled, X_test_scaled,
            pca_n, ae_latent, ae_lr, ae_dropout
        )

        # Classifier hyperparameters
        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        C = trial.suggest_float('C', 1e-3, 100.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9) if penalty == 'elasticnet' else 0.5

        # Evaluate
        val_acc, val_std, train_acc = purged_cv_score(
            X_train_feat, y, ts_groups, penalty, C, l1_ratio, n_splits=3
        )

        trial.set_user_attr('train_acc', train_acc)
        trial.set_user_attr('gap', train_acc - val_acc)

        return val_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best Purged CV accuracy: {study.best_value:.4f}")
    print(f"  Train-val gap: {study.best_trial.user_attrs['gap']:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ── 4. Build final features ──
    print("\n[4/6] Building final features with best hyperparams...")
    bp = study.best_params
    X_train_final, X_test_final = build_features(
        X_train_scaled, X_test_scaled,
        pca_n=bp['pca_n_components'],
        ae_latent=bp['ae_latent_dim'],
        ae_lr=bp['ae_lr'],
        ae_dropout=bp['ae_dropout'],
    )
    print(f"  Feature shape: Train {X_train_final.shape}, Test {X_test_final.shape}")

    # ── 5. Final evaluation (5-fold) ──
    print("\n[5/6] Final 5-fold Purged KFold evaluation...")
    l1_ratio = bp.get('l1_ratio', 0.5)
    val_acc, val_std, train_acc = purged_cv_score(
        X_train_final, y, ts_groups,
        bp['penalty'], bp['C'], l1_ratio,
        n_splits=5
    )
    print(f"  Train: {train_acc:.4f} | Val: {val_acc:.4f} (±{val_std:.4f}) | Gap: {train_acc - val_acc:.4f}")

    # ── 6. Train final model + calibrated submission ──
    print("\n[6/6] Training final model & generating submission...")
    clf = RegularizedLogistic(
        penalty=bp['penalty'],
        C=bp['C'],
        l1_ratio=l1_ratio,
    )
    clf.fit(X_train_final, y)

    preds, threshold = clf.predict_calibrated(X_test_final, y)
    target_ratio = y.mean()

    print(f"  Threshold: {threshold:.6f} (target ratio: {target_ratio:.4f})")
    print(f"  Predictions: {preds.sum()} pos / {len(preds)} total ({preds.mean()*100:.1f}%)")

    # Save submission
    submission = pd.DataFrame({'ROW_ID': test_ids, 'score': preds})
    output_file = 'submission_j4_pca_ae_logreg.csv'
    submission.to_csv(output_file, index=False)
    print(f"\n  Saved: {output_file}")

    # Top 5 trials recap
    print("\n─── Top 5 Trials ───")
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values('value', ascending=False).head(5)
    for _, row in trials_df.iterrows():
        print(f"  Trial {int(row['number']):>2}: val={row['value']:.4f}, gap={row['user_attrs_gap']:.4f}")

    print("\n" + "=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='J4 Pure Pipeline')
    parser.add_argument('--sample', action='store_true', help='Use sample data')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of Optuna trials')
    args = parser.parse_args()
    run(args)
