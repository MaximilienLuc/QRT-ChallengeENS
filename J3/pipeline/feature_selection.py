import pandas as pd
import numpy as np
import mlflow
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEvaluator:
    def __init__(self, model=None, cv=3):
        self.model = model if model else RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    def evaluate_features(self, X, y, feature_names=None):
        """
        Evaluate features using cross-validation.
        """
        if feature_names:
            X_eval = X[feature_names]
        else:
            X_eval = X
        
        scores = cross_val_score(self.model, X_eval, y, cv=self.cv, scoring='accuracy')
        mean_score = scores.mean()
        std_score = scores.std()
        
        # Fit on full data to get feature importance
        self.model.fit(X_eval, y)
        importances = pd.DataFrame({
            'feature': X_eval.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return mean_score, std_score, importances

    def check_correlation(self, X, threshold=0.95):
        """
        Identify features with correlation above threshold.
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return to_drop

def plot_feature_importance(importances, top_n=20, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.barplot(x="importance", y="feature", data=importances.head(top_n))
    plt.title(f"Top {top_n} Feature Importance")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show() # In non-interactive environments, this might error or do nothing
    plt.close()

def log_experiment_to_mlflow(run_name, params, metrics, artifacts=None):
    """
    Helper to log runs. Note: Ensure mlflow.set_experiment corresponds to valid experiment.
    """
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if artifacts:
            for artifact_path in artifacts:
                mlflow.log_artifact(artifact_path)
