"""
PCA-based feature extraction.
"""

import numpy as np
from sklearn.decomposition import PCA


class PCAFeatureExtractor:
    """
    Wrapper around sklearn PCA for consistent API.

    Usage:
        extractor = PCAFeatureExtractor(n_components=15)
        X_train_pca = extractor.fit_transform(X_train_scaled)
        X_test_pca  = extractor.transform(X_test_scaled)
    """

    def __init__(self, n_components=15):
        """
        Args:
            n_components: int or float.
                - int: exact number of components
                - float (0-1): keep enough components to explain this fraction of variance
        """
        self.n_components = n_components
        self.pca = PCA(n_components=n_components, random_state=42)

    def fit_transform(self, X):
        """Fit PCA on X and return transformed features."""
        X_pca = self.pca.fit_transform(X)
        self._log_info()
        return X_pca

    def transform(self, X):
        """Transform X using fitted PCA."""
        return self.pca.transform(X)

    def _log_info(self):
        evr = self.pca.explained_variance_ratio_
        cumsum = np.cumsum(evr)
        n = len(evr)
        print(f"  PCA: {n} components, "
              f"variance explained: {cumsum[-1]*100:.1f}%")
        print(f"    Top 5 components: {[f'{v:.1%}' for v in evr[:5]]}")

    @property
    def n_components_fitted(self):
        return self.pca.n_components_
