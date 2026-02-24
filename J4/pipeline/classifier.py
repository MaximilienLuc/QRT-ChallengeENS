"""
Regularized Logistic Regression classifier.
Supports L1, L2, and ElasticNet regularization.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score


class RegularizedLogistic:
    """
    Wrapper for sklearn logistic regression with configurable regularization.

    Args:
        penalty: 'l1', 'l2', or 'elasticnet'
        C: inverse regularization strength (smaller = stronger reg)
        l1_ratio: ElasticNet mixing (0=pure L2, 1=pure L1). Only used if penalty='elasticnet'
        max_iter: max iterations for solver convergence
    """

    def __init__(self, penalty='l2', C=1.0, l1_ratio=0.5, max_iter=1000):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.model = None
        self.threshold = 0.5

    def _build_model(self):
        if self.penalty == 'elasticnet':
            # SGDClassifier with log_loss supports elasticnet
            # alpha ≈ 1/C for SGDClassifier
            self.model = SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                alpha=1.0 / self.C,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=42,
                n_jobs=-1,
            )
        else:
            solver = 'saga' if self.penalty == 'l1' else 'lbfgs'
            self.model = LogisticRegression(
                penalty=self.penalty,
                C=self.C,
                solver=solver,
                max_iter=self.max_iter,
                random_state=42,
                n_jobs=-1,
            )

    def fit(self, X, y):
        self._build_model()
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # SGDClassifier decision_function → sigmoid
            decision = self.model.decision_function(X)
            return 1 / (1 + np.exp(-decision))

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs > self.threshold).astype(int)

    def calibrate_threshold(self, y_train):
        """
        Set threshold to match training positive ratio.
        Call after predict_proba on test data.
        """
        self._target_ratio = y_train.mean()
        return self._target_ratio

    def predict_calibrated(self, X, y_train):
        """
        Predict with calibrated threshold to match training distribution.
        """
        probs = self.predict_proba(X)
        target_ratio = y_train.mean() if hasattr(y_train, 'mean') else np.mean(y_train)
        sorted_probs = np.sort(probs)[::-1]
        n_positive = int(len(probs) * target_ratio)
        threshold = sorted_probs[min(n_positive, len(sorted_probs) - 1)]
        preds = (probs > threshold).astype(int)
        self.threshold = threshold
        return preds, threshold
