"""
PyTorch Autoencoder for unsupervised feature extraction.
Learns a compressed latent representation of the input features.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """
    Symmetric autoencoder with BatchNorm and Dropout.

    Architecture:
        Input(d) → 128 → 64 → latent_dim → 64 → 128 → d
    """

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
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        """Extract latent features only."""
        return self.encoder(x)


class AutoencoderFeatureExtractor:
    """
    High-level wrapper for training an autoencoder and extracting features.

    Usage:
        ae = AutoencoderFeatureExtractor(input_dim=41, latent_dim=16, lr=1e-3)
        ae.fit(X_train_scaled)
        X_train_ae = ae.transform(X_train_scaled)
        X_test_ae  = ae.transform(X_test_scaled)
    """

    def __init__(self, input_dim, latent_dim=16, lr=1e-3, dropout=0.3,
                 n_epochs=100, batch_size=512, patience=10, verbose=True):
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
        """
        Train the autoencoder on X (ndarray, already scaled).
        Uses early stopping on validation reconstruction loss.
        """
        # Split into train/val
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
            # Train
            self.model.train()
            train_loss = 0
            for (batch,) in train_dl:
                batch = batch.to(self.device)
                x_hat = self.model(batch)
                loss = criterion(x_hat, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(batch)
            train_loss /= len(X_train)

            # Validate
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for (batch,) in val_dl:
                    batch = batch.to(self.device)
                    x_hat = self.model(batch)
                    val_loss += criterion(x_hat, batch).item() * len(batch)
            val_loss /= len(X_val)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"    Epoch {epoch+1:3d}/{self.n_epochs}: "
                      f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                      f"{' *' if patience_counter == 0 else ''}")

            if patience_counter >= self.patience:
                if self.verbose:
                    print(f"    Early stopping at epoch {epoch+1}")
                break

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.model.eval()
        if self.verbose:
            print(f"  Autoencoder: latent_dim={self.latent_dim}, "
                  f"best_val_loss={best_val_loss:.6f}")

        return self

    def transform(self, X):
        """Extract latent features from X (ndarray)."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            # Process in batches to avoid OOM
            latent_parts = []
            for i in range(0, len(X), self.batch_size * 4):
                batch = X_tensor[i:i + self.batch_size * 4]
                z = self.model.encode(batch)
                latent_parts.append(z.cpu().numpy())
        return np.concatenate(latent_parts, axis=0)

    def fit_transform(self, X, val_frac=0.1):
        """Fit and transform in one call."""
        self.fit(X, val_frac=val_frac)
        return self.transform(X)
