"""
J4 Pipeline — Colab Version
===========================
This script combines all the J4 components (PCA, Autoencoder, Logistic Regression)
into a single file that can easily be run on Google Colab with a GPU.

Requirements:
- Ensure the data files are accessible at the specified `data_dir` path.
- This script will save OOF probabilities and the final submission file.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings('ignore')

# =====================================================================
# 1. UTILS (Purged KFold)
# =====================================================================

class PurgedKFold:
    """Time-series cross validation with embargo/purging."""
    def __init__(self, n_splits=5, purge_pct=0.02):
        self.n_splits = n_splits
        self.purge_pct = purge_pct

    def split(self, X, y=None, groups=None):
        unique_times = np.sort(np.unique(groups))
        n_times = len(unique_times)
        purge_size = int(n_times * self.purge_pct)
        fold_size = n_times // self.n_splits

        for i in range(self.n_splits):
            val_start = i * fold_size
            val_end = (i + 1) * fold_size if i < self.n_splits - 1 else n_times

            val_times = set(unique_times[val_start:val_end])

            purge_start = max(0, val_start - purge_size)
            purge_end = min(n_times, val_end + purge_size)
            excluded_times = set(unique_times[purge_start:purge_end])
            train_times = set(unique_times) - excluded_times

            train_idx = np.where(np.isin(groups, list(train_times)))[0]
            val_idx = np.where(np.isin(groups, list(val_times)))[0]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


# =====================================================================
# 2. FEATURE EXTRACTION (PCA & Autoencoder)
# =====================================================================

class PCAFeatureExtractor:
    def __init__(self, n_components=15):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit_transform(self, X):
        return self.pca.fit_transform(X)

    def transform(self, X):
        return self.pca.transform(X)

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=16, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            nn.Linear(64, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x):
        return self.encoder(x)

class AutoencoderFeatureExtractor:
    def __init__(self, input_dim, latent_dim=16, lr=1e-3, dropout=0.3,
                 n_epochs=100, batch_size=512, patience=10, verbose=False):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.lr = lr
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.verbose = verbose

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Autoencoder(input_dim, latent_dim, dropout).to(self.device)

    def fit(self, X, val_frac=0.1):
        n = len(X)
        n_val = int(n * val_frac)
        indices = np.random.RandomState(42).permutation(n)
        X_train = X[indices[n_val:]]
        X_val = X[indices[:n_val]]

        train_ds = TensorDataset(torch.FloatTensor(X_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val))
        train_dl = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size * 2, shuffle=False)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        for epoch in range(self.n_epochs):
            self.model.train()
            for (batch,) in train_dl:
                batch = batch.to(self.device)
                x_hat = self.model(batch)
                loss = criterion(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for (batch,) in val_dl:
                    batch = batch.to(self.device)
                    x_hat = self.model(batch)
                    val_loss += criterion(x_hat, batch).item() * len(batch)
            val_loss /= len(X_val)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return self

    def transform(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            latent_parts = []
            for i in range(0, len(X), self.batch_size * 4):
                batch = X_tensor[i:i + self.batch_size * 4]
                z = self.model.encode(batch)
                latent_parts.append(z.cpu().numpy())
        return np.concatenate(latent_parts, axis=0)

    def fit_transform(self, X, val_frac=0.1):
        self.fit(X, val_frac=val_frac)
        return self.transform(X)


# =====================================================================
# 3. CLASSIFIER (Logistic Regression)
# =====================================================================

class RegularizedLogistic:
    def __init__(self, penalty='l2', C=1.0, l1_ratio=0.5, max_iter=1000):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.model = None
        self.threshold = 0.5

    def _build_model(self):
        if self.penalty == 'elasticnet':
            self.model = SGDClassifier(
                loss='log_loss', penalty='elasticnet', alpha=1.0 / self.C,
                l1_ratio=self.l1_ratio, max_iter=self.max_iter, random_state=42, n_jobs=-1,
            )
        else:
            solver = 'saga' if self.penalty == 'l1' else 'lbfgs'
            self.model = LogisticRegression(
                penalty=self.penalty, C=self.C, solver=solver,
                max_iter=self.max_iter, random_state=42, n_jobs=-1,
            )

    def fit(self, X, y):
        self._build_model()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            decision = self.model.decision_function(X)
            return 1 / (1 + np.exp(-decision))

    def predict(self, X):
        return (self.predict_proba(X) > self.threshold).astype(int)

    def predict_calibrated(self, X, y_train):
        probs = self.predict_proba(X)
        target_ratio = y_train.mean() if hasattr(y_train, 'mean') else np.mean(y_train)
        sorted_probs = np.sort(probs)[::-1]
        n_positive = int(len(probs) * target_ratio)
        threshold = sorted_probs[min(n_positive, len(sorted_probs) - 1)]
        preds = (probs > threshold).astype(int)
        self.threshold = threshold
        return preds, threshold


# =====================================================================
# 4. MAIN PIPELINE
# =====================================================================

def build_features(X_train_scaled, X_test_scaled, pca_n, ae_latent, ae_lr, ae_dropout):
    input_dim = X_train_scaled.shape[1]
    
    pca = PCAFeatureExtractor(n_components=pca_n)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    ae = AutoencoderFeatureExtractor(
        input_dim=input_dim, latent_dim=ae_latent, lr=ae_lr, dropout=ae_dropout,
    )
    X_train_ae = ae.fit_transform(X_train_scaled)
    X_test_ae = ae.transform(X_test_scaled)

    X_train_combined = np.hstack([X_train_pca, X_train_ae])
    X_test_combined = np.hstack([X_test_pca, X_test_ae])

    return X_train_combined, X_test_combined


def purged_cv_score(X_combined, y, ts_groups, penalty, C, l1_ratio, n_splits=3):
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=0.02)
    val_scores, train_scores = [], []
    oof_preds = np.zeros(len(X_combined))

    for train_idx, val_idx in pkf.split(X_combined, groups=ts_groups):
        clf = RegularizedLogistic(penalty=penalty, C=C, l1_ratio=l1_ratio)
        
        y_train_fold = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
        y_val_fold = y.iloc[val_idx] if hasattr(y, 'iloc') else y[val_idx]

        clf.fit(X_combined[train_idx], y_train_fold)

        val_preds = clf.predict(X_combined[val_idx])
        val_scores.append(accuracy_score(y_val_fold, val_preds))
        train_scores.append(accuracy_score(y_train_fold, clf.predict(X_combined[train_idx])))
        
        oof_preds[val_idx] = clf.predict_proba(X_combined[val_idx])

    return np.mean(val_scores), np.mean(train_scores), oof_preds


def process_data(data_dir='Data'):
    """Loads, imputes, and scales the data."""
    print("Loading data...")
    X_train_raw = pd.read_csv(f'{data_dir}/X_train.csv')
    y_train_raw = pd.read_csv(f'{data_dir}/y_train.csv')
    X_test_raw = pd.read_csv(f'{data_dir}/X_test.csv')

    train_df = X_train_raw.merge(y_train_raw, on='ROW_ID')
    y = (train_df['target'] > 0).astype(int)

    X_train = train_df.drop(columns=['target', 'ROW_ID'])
    X_test = X_test_raw.drop(columns=['ROW_ID'])
    test_ids = X_test_raw['ROW_ID']
    ts_groups = X_train['TS'].str.extract(r'(\d+)')[0].astype(int).values

    X_train = X_train.drop(columns=['TS'])
    if 'TS' in X_test.columns:
        X_test = X_test.drop(columns=['TS'])

    non_numeric = X_train.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X_train = X_train.drop(columns=non_numeric)
        X_test = X_test.drop(columns=[c for c in non_numeric if c in X_test.columns])

    print("Imputing and scaling...")
    imputer = SimpleImputer(strategy='mean')
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    return X_train_scaled, X_test_scaled, y, ts_groups, test_ids


def run(data_dir='Data', n_trials=30):
    print("=" * 60)
    print("  J4 COLAB PIPELINE: PCA + AUTOENCODER + LOGREG")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  PyTorch Device: {device}")

    X_train_scaled, X_test_scaled, y, ts_groups, test_ids = process_data(data_dir)
    input_dim = X_train_scaled.shape[1]

    # --- Optuna ---
    print(f"\nOptuna optimization ({n_trials} trials)...")

    def objective(trial):
        pca_n = trial.suggest_int('pca_n_components', 5, min(input_dim, 35))
        ae_latent = trial.suggest_int('ae_latent_dim', 8, 32)
        ae_lr = trial.suggest_float('ae_lr', 1e-4, 1e-2, log=True)
        ae_dropout = trial.suggest_float('ae_dropout', 0.1, 0.5)

        X_train_feat, _ = build_features(
            X_train_scaled, X_test_scaled, pca_n, ae_latent, ae_lr, ae_dropout
        )

        penalty = trial.suggest_categorical('penalty', ['l1', 'l2', 'elasticnet'])
        C = trial.suggest_float('C', 1e-3, 100.0, log=True)
        l1_ratio = trial.suggest_float('l1_ratio', 0.1, 0.9) if penalty == 'elasticnet' else 0.5

        val_acc, _, _ = purged_cv_score(
            X_train_feat, y, ts_groups, penalty, C, l1_ratio, n_splits=3
        )
        return val_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n  Best Purged CV accuracy: {study.best_value:.4f}")
    bp = study.best_params

    # --- Build final features ---
    print("\nBuilding final features with best hyperparams...")
    X_train_final, X_test_final = build_features(
        X_train_scaled, X_test_scaled,
        pca_n=bp['pca_n_components'], ae_latent=bp['ae_latent_dim'],
        ae_lr=bp['ae_lr'], ae_dropout=bp['ae_dropout'],
    )

    # --- Final evaluation to get OOF ---
    print("\nFinal 5-fold evaluation for OOF generation...")
    l1_ratio = bp.get('l1_ratio', 0.5)
    val_acc, train_acc, oof_preds = purged_cv_score(
        X_train_final, y, ts_groups, bp['penalty'], bp['C'], l1_ratio, n_splits=5
    )
    
    np.save('oof_j4_pca_ae_logreg.npy', oof_preds)
    print(f"  Saved OOF probabilities -> 'oof_j4_pca_ae_logreg.npy'")

    # --- Train final model ---
    print("\nTraining final model & generating submission...")
    clf = RegularizedLogistic(penalty=bp['penalty'], C=bp['C'], l1_ratio=l1_ratio)
    clf.fit(X_train_final, y)

    preds, threshold = clf.predict_calibrated(X_test_final, y)
    test_probs = clf.predict_proba(X_test_final)

    # Save absolute probabilities for greedy blending
    np.save('test_j4_pca_ae_logreg.npy', test_probs)
    print(f"  Saved TEST probabilities -> 'test_j4_pca_ae_logreg.npy'")

    submission = pd.DataFrame({'ROW_ID': test_ids, 'prediction': preds})
    output_file = 'submission_j4_pca_ae_logreg.csv'
    submission.to_csv(output_file, index=False)
    print(f"  Saved Submission -> '{output_file}'")

if __name__ == "__main__":
    # If using Google Drive in Colab, change data_dir to your drive path
    run(data_dir='Data', n_trials=30)
