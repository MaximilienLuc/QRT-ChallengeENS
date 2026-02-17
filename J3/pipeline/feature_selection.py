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

    def check_drift(self, X_train, X_test, threshold=0.70):
        """
        Adversarial Validation to detect drifting features.
        Returns list of features to drop.
        """
        import lightgbm as lgb
        from sklearn.metrics import roc_auc_score
        
        # 1. Label Train=0, Test=1
        X_train['is_test'] = 0
        X_test['is_test'] = 1
        
        # 2. Combine
        X_adv = pd.concat([X_train, X_test], axis=0)
        y_adv = X_adv['is_test']
        X_adv = X_adv.drop(columns=['is_test'])
        
        # Clean up original DFs
        X_train.drop(columns=['is_test'], inplace=True)
        X_test.drop(columns=['is_test'], inplace=True)
        
        # 3. Train Classifier
        # Simple/fast params for drift detection
        clf = lgb.LGBMClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=42,
            n_jobs=-1,
            verbosity=-1
        )
        
        # Cross-val to get stable AUC
        cv_scores = cross_val_score(clf, X_adv, y_adv, cv=3, scoring='roc_auc')
        mean_auc = cv_scores.mean()
        
        print(f"Adversarial Validation AUC: {mean_auc:.4f}")
        
        to_drop = []
        if mean_auc > threshold:
            print(f"Drift detected (AUC > {threshold}). Identifying culprits...")
            
            # Fit on full data for importance
            clf.fit(X_adv, y_adv)
            
            importances = pd.DataFrame({
                'feature': X_adv.columns,
                'importance': clf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            # Heuristic: Drop top contributing features until AUC drops?
            # Or just drop top N that make up 50% of importance?
            # Here: Drop top 20 features for now as requested user strategy "remove top 20"
            to_drop = importances.head(20)['feature'].tolist()
            
            print(f"Top 5 drifting features: {to_drop[:5]}")
            
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
