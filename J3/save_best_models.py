"""
Save Best Models — LGBM + XGBoost + Meta-learner
---
Réentraîne les deux modèles sur TOUT le train avec les meilleurs params
trouvés par Optuna, puis sauvegarde avec joblib.

Sauvegarde dans models/ :
  lgbm_best.pkl      → LGBMClassifier full train
  xgb_best.pkl       → XGBClassifier full train
  meta_lr.pkl        → LogReg méta-learner (entraîné sur OOF)
  feature_cols.json  → liste des features utilisées
  best_params.json   → meilleurs params des deux modèles
"""

import os, sys, json, warnings
import pandas as pd
import numpy as np
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("."))  # même pattern que les autres scripts

from pipeline.feature_engineering import (
    RawFeatureGenerator,
    RollingStatFeatureGenerator,
    MomentumGenerator,
    VolatilityRatioGenerator,
    ShortTermInteractionGenerator,
)
from pipeline.feature_selection import FeatureEvaluator

# ─── Best params issus du run Optuna ─────────────────────────────────────────

BEST_LGBM = {
    "objective": "binary",
    "verbosity": -1,
    "boosting_type": "gbdt",
    "n_estimators": 258,
    "learning_rate": 0.008217026556199275,
    "num_leaves": 29,
    "max_depth": 7,
    "min_child_samples": 108,
    "subsample": 0.6070300501614984,
    "colsample_bytree": 0.30985950229003156,
    "bagging_freq": 4,
    "reg_alpha": 0.1448322683470355,
    "reg_lambda": 9.695129279672335,
    "min_split_gain": 1.5387420395409699,
    "path_smooth": 2.511194418285826,
}

BEST_XGB = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "verbosity": 0,
    "n_estimators": 513,
    "learning_rate": 0.005344132196309563,
    "max_depth": 5,
    "min_child_weight": 60,
    "gamma": 3.244198836221166,
    "max_delta_step": 5,
    "subsample": 0.6380549794383925,
    "colsample_bytree": 0.4055778764905851,
    "colsample_bylevel": 0.6950467824859545,
    "reg_alpha": 0.09927636876729332,
    "reg_lambda": 3.0276198791610587,
}


# ─── Purged KFold (pour régénérer les OOF du méta-learner) ───────────────────

class PurgedKFold:
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
            excluded = set(unique_times[purge_start:purge_end])
            train_times = set(unique_times) - excluded
            tr_idx = np.where(np.isin(groups, list(train_times)))[0]
            val_idx = np.where(np.isin(groups, list(val_times)))[0]
            if len(tr_idx) > 0 and len(val_idx) > 0:
                yield tr_idx, val_idx


# ─── Feature Engineering ─────────────────────────────────────────────────────

def make_generators():
    return [
        RawFeatureGenerator(
            cols=[f"RET_{i}" for i in range(1, 21)]
            + [f"SIGNED_VOLUME_{i}" for i in range(1, 21)]
            + ["MEDIAN_DAILY_TURNOVER"]
        ),
        RollingStatFeatureGenerator(
            cols=[f"RET_{i}" for i in range(1, 21)],
            windows=[5, 10, 20],
            operations=["mean", "std", "min", "max"],
        ),
        MomentumGenerator(windows=[(5, 20), (1, 5), (1, 20)]),
        VolatilityRatioGenerator(windows=[(5, 20)]),
        ShortTermInteractionGenerator(max_lag=10),
    ]


def build_features(X, y=None, generators=None, fit=True):
    X_feat = pd.DataFrame(index=X.index)
    for gen in generators:
        if fit and hasattr(gen, "fit"):
            gen.fit(X, y)
        X_feat = pd.concat([X_feat, gen.transform(X)], axis=1)
    return X_feat.loc[:, ~X_feat.columns.duplicated()]


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    print("=" * 65)
    print("  SAVE BEST MODELS — LGBM + XGBoost + Meta-learner")
    print("=" * 65)

    # Créer le dossier models/
    os.makedirs("models", exist_ok=True)
    print("\n  Output dir: J3/models/")

    # ── 1. Load & feature engineering ────────────────────────────────────────
    print("\n[1/5] Load + feature engineering...")
    X_train_full = pd.read_csv("Data/X_train.csv")
    y_train_full = pd.read_csv("Data/y_train.csv")
    X_test_full  = pd.read_csv("Data/X_test.csv")

    train_df = X_train_full.merge(y_train_full, on="ROW_ID")
    y_full   = (train_df["target"] > 0).astype(int)
    X_full   = train_df.drop(columns=["target", "ROW_ID"])
    X_test   = X_test_full.drop(columns=["ROW_ID"])
    ts_full  = X_full["TS"].str.extract(r"(\d+)")[0].astype(int).values

    generators  = make_generators()
    X_feat      = build_features(X_full, y_full, generators, fit=True)
    X_test_feat = build_features(X_test, generators=generators, fit=False)

    # ── 2. Feature selection (même pipeline que le run Optuna) ───────────────
    print("[2/5] Feature selection...")
    evaluator = FeatureEvaluator(cv=3)
    drifting  = evaluator.check_drift(X_feat, X_test_feat, threshold=0.60)
    X_feat      = X_feat.drop(columns=drifting)
    X_test_feat = X_test_feat.drop(columns=drifting)
    high_corr   = evaluator.check_correlation(X_feat, threshold=0.95)
    X_feat      = X_feat.drop(columns=high_corr)
    X_test_feat = X_test_feat.drop(columns=high_corr)
    feature_cols = list(X_feat.columns)
    print(f"  {len(feature_cols)} features retenues")

    # ── 3. Entraîner sur TOUT le train ────────────────────────────────────────
    print("\n[3/5] Entraînement full-train...")

    lgbm_full = lgb.LGBMClassifier(**BEST_LGBM, random_state=42, n_jobs=-1)
    lgbm_full.fit(X_feat, y_full)
    train_acc_lgbm = accuracy_score(y_full, lgbm_full.predict(X_feat))
    print(f"  LGBM  — train acc: {train_acc_lgbm:.4f}")

    xgb_full = xgb.XGBClassifier(**BEST_XGB, random_state=42, n_jobs=-1)
    xgb_full.fit(X_feat, y_full)
    train_acc_xgb = accuracy_score(y_full, xgb_full.predict(X_feat))
    print(f"  XGBoost — train acc: {train_acc_xgb:.4f}")

    # ── 4. Régénérer les OOF pour le méta-learner ────────────────────────────
    print("\n[4/5] OOF purged (5-fold) pour méta-learner...")
    pkf = PurgedKFold(n_splits=5, purge_pct=0.02)
    N = len(y_full)
    oof_lgbm = np.zeros(N)
    oof_xgb  = np.zeros(N)

    for fold, (tr_idx, val_idx) in enumerate(pkf.split(X_feat, groups=ts_full)):
        m_l = lgb.LGBMClassifier(**BEST_LGBM, random_state=42 + fold, n_jobs=-1)
        m_l.fit(X_feat.iloc[tr_idx], y_full.iloc[tr_idx])
        oof_lgbm[val_idx] = m_l.predict_proba(X_feat.iloc[val_idx])[:, 1]

        m_x = xgb.XGBClassifier(**BEST_XGB, random_state=42 + fold, n_jobs=-1)
        m_x.fit(X_feat.iloc[tr_idx], y_full.iloc[tr_idx])
        oof_xgb[val_idx] = m_x.predict_proba(X_feat.iloc[val_idx])[:, 1]

        l_acc = accuracy_score(y_full.iloc[val_idx], (oof_lgbm[val_idx] > 0.5).astype(int))
        x_acc = accuracy_score(y_full.iloc[val_idx], (oof_xgb[val_idx]  > 0.5).astype(int))
        print(f"  Fold {fold+1}/5 — LGBM: {l_acc:.4f} | XGB: {x_acc:.4f}")

    meta_train = np.column_stack([oof_lgbm, oof_xgb])
    meta_lr = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    meta_lr.fit(meta_train, y_full)
    oof_stack_acc = accuracy_score(y_full, meta_lr.predict(meta_train))
    print(f"\n  OOF LGBM: {accuracy_score(y_full,(oof_lgbm>0.5).astype(int)):.4f} | "
          f"OOF XGB: {accuracy_score(y_full,(oof_xgb>0.5).astype(int)):.4f} | "
          f"Stack: {oof_stack_acc:.4f}")
    print(f"  Meta-lr: w_LGBM={meta_lr.coef_[0][0]:.4f}  w_XGB={meta_lr.coef_[0][1]:.4f}")

    # ── 5. Sauvegarde ─────────────────────────────────────────────────────────
    print("\n[5/5] Sauvegarde...")

    joblib.dump(lgbm_full, "models/lgbm_best.pkl")
    print("  ✓ J3/models/lgbm_best.pkl")

    joblib.dump(xgb_full, "models/xgb_best.pkl")
    print("  ✓ J3/models/xgb_best.pkl")

    joblib.dump(meta_lr, "models/meta_lr.pkl")
    print("  ✓ J3/models/meta_lr.pkl")

    with open("models/feature_cols.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    print("  ✓ J3/models/feature_cols.json")

    with open("models/best_params.json", "w") as f:
        json.dump({"lgbm": BEST_LGBM, "xgb": BEST_XGB}, f, indent=2)
    print("  ✓ J3/models/best_params.json")

    # Tailles des fichiers
    print("\n  Tailles :")
    for fname in ["lgbm_best.pkl", "xgb_best.pkl", "meta_lr.pkl",
                  "feature_cols.json", "best_params.json"]:
        fpath = f"models/{fname}"
        size_kb = os.path.getsize(fpath) / 1024
        print(f"    {fname:<22} {size_kb:>8.1f} KB")

    print("\n" + "=" * 65)
    print("  Modèles sauvegardés dans J3/models/")
    print("  Utilisation :")
    print("    import joblib")
    print("    lgbm = joblib.load('models/lgbm_best.pkl')")
    print("    xgb  = joblib.load('models/xgb_best.pkl')")
    print("    meta = joblib.load('models/meta_lr.pkl')")
    print("=" * 65)


if __name__ == "__main__":
    run()
