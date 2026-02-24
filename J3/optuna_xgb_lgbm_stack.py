"""
Big Fat Optuna: XGBoost + LGBM + Purged Stacking
---
- 60-trial Optuna LGBM  : search space anti-overfit + pénalité de gap
- 60-trial Optuna XGBoost: idem, search space propre XGB
- Stacking via Purged KFold OOF (5-fold, purge 2%)
- Adversarial validation drift check (threshold 0.60)
- Méta-learner : LogisticRegression(C=0.1) — 2 features only, très régularisé
- Threshold calibré sur le ratio positif de train
"""

import pandas as pd
import numpy as np
import sys, os, json, warnings
import joblib
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
sys.path.append(os.path.abspath("."))

from pipeline.feature_engineering import (
    RawFeatureGenerator,
    RollingStatFeatureGenerator,
    MomentumGenerator,
    VolatilityRatioGenerator,
    ShortTermInteractionGenerator,
)
from pipeline.feature_selection import FeatureEvaluator


# ─── Purged K-Fold ────────────────────────────────────────────────────────────

class PurgedKFold:
    """
    Time-aware K-Fold.
    Validation = contiguous slice of time periods.
    Purge = buffer zone autour du val exclu du train → évite la fuite temporelle.
    """
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


def purged_cv_score(model_fn, X, y, ts_groups, n_splits=3):
    """
    CV purged générique.
    model_fn() → sklearn-compatible estimator frais à chaque fold.
    Retourne (val_mean, val_std, train_mean).
    """
    pkf = PurgedKFold(n_splits=n_splits, purge_pct=0.02)
    val_scores, train_scores = [], []

    for tr_idx, val_idx in pkf.split(X, groups=ts_groups):
        model = model_fn()
        model.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        val_scores.append(
            accuracy_score(y.iloc[val_idx], model.predict(X.iloc[val_idx]))
        )
        train_scores.append(
            accuracy_score(y.iloc[tr_idx], model.predict(X.iloc[tr_idx]))
        )

    return np.mean(val_scores), np.std(val_scores), np.mean(train_scores)


# ─── Feature Engineering ──────────────────────────────────────────────────────

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
    print("=" * 70)
    print("  BIG FAT OPTUNA : XGBoost + LGBM + Purged Stacking")
    print("=" * 70)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print("\n[1/7] Loading data...")
    X_train_full = pd.read_csv("Data/X_train.csv")
    y_train_full = pd.read_csv("Data/y_train.csv")
    X_test_full = pd.read_csv("Data/X_test.csv")

    train_df = X_train_full.merge(y_train_full, on="ROW_ID")
    y_full = (train_df["target"] > 0).astype(int)
    X_full = train_df.drop(columns=["target", "ROW_ID"])
    X_test = X_test_full.drop(columns=["ROW_ID"])
    ts_full = X_full["TS"].str.extract(r"(\d+)")[0].astype(int).values

    print(f"  Train: {X_full.shape} | Test: {X_test.shape}")

    # ── 2. Feature Engineering ────────────────────────────────────────────────
    print("\n[2/7] Feature engineering...")
    generators = make_generators()
    X_feat = build_features(X_full, y_full, generators, fit=True)
    X_test_feat = build_features(X_test, generators=generators, fit=False)
    print(f"  {X_feat.shape[1]} features générées")

    # ── 3. Feature Selection ──────────────────────────────────────────────────
    print("\n[3/7] Feature selection (adversarial + corrélation)...")
    evaluator = FeatureEvaluator(cv=3)

    drifting = evaluator.check_drift(X_feat, X_test_feat, threshold=0.60)
    X_feat = X_feat.drop(columns=drifting)
    X_test_feat = X_test_feat.drop(columns=drifting)
    print(f"  Dropped {len(drifting)} drifting features")

    high_corr = evaluator.check_correlation(X_feat, threshold=0.95)
    X_feat = X_feat.drop(columns=high_corr)
    X_test_feat = X_test_feat.drop(columns=high_corr)
    print(f"  Dropped {len(high_corr)} correlated → {X_feat.shape[1]} features restantes")

    N = len(y_full)

    # ── 4. Optuna LGBM ────────────────────────────────────────────────────────
    print("\n[4/7] Optuna LGBM — 60 trials, anti-overfit...")
    print("  Search space : num_leaves≤48, max_depth≤7, min_child_samples≥50,")
    print("                 strong L1/L2, path_smooth, pénalité gap > 3%")

    def objective_lgbm(trial):
        p = {
            "objective": "binary",
            "verbosity": -1,
            "boosting_type": "gbdt",
            # Taille des arbres : petits pour éviter l'overfit
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 48),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            # Régularisation via min samples par feuille
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 400),
            # Sous-échantillonnage agressif
            "subsample": trial.suggest_float("subsample", 0.4, 0.75),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.65),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
            # L1 / L2 forts
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 100.0, log=True),
            # Gain minimal pour splitter
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 2.0),
            # Lissage du chemin (régularisation supplémentaire LGB 3+)
            "path_smooth": trial.suggest_float("path_smooth", 0.0, 10.0),
        }

        val_acc, _, train_acc = purged_cv_score(
            lambda: lgb.LGBMClassifier(**p, random_state=42, n_jobs=-1),
            X_feat, y_full, ts_full, n_splits=3,
        )
        gap = train_acc - val_acc
        trial.set_user_attr("train_acc", train_acc)
        trial.set_user_attr("gap", gap)
        # Pénaliser l'overfit : on tolère 3% de gap, coût 0.2x au-delà
        return val_acc - 0.2 * max(0.0, gap - 0.03)

    study_lgbm = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_lgbm.optimize(objective_lgbm, n_trials=60, show_progress_bar=True)

    best_lgbm_p = study_lgbm.best_params.copy()
    best_lgbm_p.update({"objective": "binary", "verbosity": -1, "boosting_type": "gbdt"})

    t_lgbm = study_lgbm.best_trial
    print(f"\n  LGBM best trial #{t_lgbm.number}")
    print(f"  Score pénalisé : {study_lgbm.best_value:.4f}  |  gap : {t_lgbm.user_attrs['gap']:.4f}")
    print("  Params :")
    for k, v in best_lgbm_p.items():
        print(f"    {k}: {v}")

    # Re-eval 5-fold
    val_lgbm, std_lgbm, tr_lgbm = purged_cv_score(
        lambda: lgb.LGBMClassifier(**best_lgbm_p, random_state=42, n_jobs=-1),
        X_feat, y_full, ts_full, n_splits=5,
    )
    gap_lgbm = tr_lgbm - val_lgbm
    print(f"\n  5-fold purged LGBM → train={tr_lgbm:.4f} | val={val_lgbm:.4f} ± {std_lgbm:.4f} | gap={gap_lgbm:.4f}")

    # Top 5 LGBM trials
    print("\n  Top 5 trials LGBM :")
    lgbm_df = study_lgbm.trials_dataframe().sort_values("value", ascending=False).head(5)
    for _, row in lgbm_df.iterrows():
        print(f"    #{int(row['number']):>2}: score={row['value']:.4f}  train={row['user_attrs_train_acc']:.4f}  gap={row['user_attrs_gap']:.4f}")

    # ── 5. Optuna XGBoost ─────────────────────────────────────────────────────
    print("\n[5/7] Optuna XGBoost — 60 trials, anti-overfit...")
    print("  Search space : max_depth≤6, min_child_weight≥10, gamma, max_delta_step,")
    print("                 strong alpha/lambda, pénalité gap > 3%")

    def objective_xgb(trial):
        p = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "verbosity": 0,
            # Taille des arbres
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            # Régularisation XGB spécifique
            "min_child_weight": trial.suggest_int("min_child_weight", 10, 100),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "max_delta_step": trial.suggest_int("max_delta_step", 0, 5),
            # Sous-échantillonnage
            "subsample": trial.suggest_float("subsample", 0.4, 0.75),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 0.65),
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
            # L1 / L2 forts
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 100.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 100.0, log=True),
        }

        val_acc, _, train_acc = purged_cv_score(
            lambda: xgb.XGBClassifier(**p, random_state=42, n_jobs=-1),
            X_feat, y_full, ts_full, n_splits=3,
        )
        gap = train_acc - val_acc
        trial.set_user_attr("train_acc", train_acc)
        trial.set_user_attr("gap", gap)
        return val_acc - 0.2 * max(0.0, gap - 0.03)

    study_xgb = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study_xgb.optimize(objective_xgb, n_trials=60, show_progress_bar=True)

    best_xgb_p = study_xgb.best_params.copy()
    best_xgb_p.update({
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "verbosity": 0,
    })

    t_xgb = study_xgb.best_trial
    print(f"\n  XGB best trial #{t_xgb.number}")
    print(f"  Score pénalisé : {study_xgb.best_value:.4f}  |  gap : {t_xgb.user_attrs['gap']:.4f}")
    print("  Params :")
    for k, v in best_xgb_p.items():
        print(f"    {k}: {v}")

    # Re-eval 5-fold
    val_xgb, std_xgb, tr_xgb = purged_cv_score(
        lambda: xgb.XGBClassifier(**best_xgb_p, random_state=42, n_jobs=-1),
        X_feat, y_full, ts_full, n_splits=5,
    )
    gap_xgb = tr_xgb - val_xgb
    print(f"\n  5-fold purged XGB  → train={tr_xgb:.4f} | val={val_xgb:.4f} ± {std_xgb:.4f} | gap={gap_xgb:.4f}")

    # Top 5 XGB trials
    print("\n  Top 5 trials XGB :")
    xgb_df = study_xgb.trials_dataframe().sort_values("value", ascending=False).head(5)
    for _, row in xgb_df.iterrows():
        print(f"    #{int(row['number']):>2}: score={row['value']:.4f}  train={row['user_attrs_train_acc']:.4f}  gap={row['user_attrs_gap']:.4f}")

    # ── 6. Stacking OOF via Purged KFold ──────────────────────────────────────
    print("\n[6/7] Stacking — OOF via Purged KFold (5-fold)...")
    print("  Pour chaque fold : train LGBM + XGB → OOF proba")
    print("  Test predictions : moyenne des K modèles fold (pas de refit global → moins de fuite)")

    pkf_stack = PurgedKFold(n_splits=5, purge_pct=0.02)

    oof_lgbm = np.zeros(N)
    oof_xgb = np.zeros(N)
    oof_covered = np.zeros(N, dtype=bool)  # sanity check : tous les samples couverts ?

    test_lgbm_folds = []
    test_xgb_folds = []

    for fold, (tr_idx, val_idx) in enumerate(pkf_stack.split(X_feat, groups=ts_full)):
        X_tr = X_feat.iloc[tr_idx]
        X_val = X_feat.iloc[val_idx]
        y_tr = y_full.iloc[tr_idx]
        y_val = y_full.iloc[val_idx]

        # LGBM
        m_lgbm = lgb.LGBMClassifier(**best_lgbm_p, random_state=42 + fold, n_jobs=-1)
        m_lgbm.fit(X_tr, y_tr)
        oof_lgbm[val_idx] = m_lgbm.predict_proba(X_val)[:, 1]
        test_lgbm_folds.append(m_lgbm.predict_proba(X_test_feat)[:, 1])

        # XGB
        m_xgb = xgb.XGBClassifier(**best_xgb_p, random_state=42 + fold, n_jobs=-1)
        m_xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = m_xgb.predict_proba(X_val)[:, 1]
        test_xgb_folds.append(m_xgb.predict_proba(X_test_feat)[:, 1])

        oof_covered[val_idx] = True

        lgbm_fold_acc = accuracy_score(y_val, (oof_lgbm[val_idx] > 0.5).astype(int))
        xgb_fold_acc = accuracy_score(y_val, (oof_xgb[val_idx] > 0.5).astype(int))
        print(f"  Fold {fold+1}/5 — LGBM: {lgbm_fold_acc:.4f} | XGB: {xgb_fold_acc:.4f}")

    n_uncovered = (~oof_covered).sum()
    if n_uncovered > 0:
        print(f"  ⚠️  {n_uncovered} samples non couverts (purge boundaries) — imputés à 0.5")
        oof_lgbm[~oof_covered] = 0.5
        oof_xgb[~oof_covered] = 0.5

    # OOF individuels
    acc_oof_lgbm = accuracy_score(y_full, (oof_lgbm > 0.5).astype(int))
    acc_oof_xgb = accuracy_score(y_full, (oof_xgb > 0.5).astype(int))

    # Corrélation entre les deux sets OOF (diversité du stacking)
    oof_corr = np.corrcoef(oof_lgbm, oof_xgb)[0, 1]

    print(f"\n  OOF accuracy — LGBM: {acc_oof_lgbm:.4f} | XGB: {acc_oof_xgb:.4f}")
    print(f"  Corrélation OOF LGBM↔XGB : {oof_corr:.4f}  (plus bas = plus diversifié)")

    # Test proba : moyenne des folds
    test_lgbm_avg = np.mean(test_lgbm_folds, axis=0)
    test_xgb_avg = np.mean(test_xgb_folds, axis=0)

    # ── Méta-learner : LogReg ─────────────────────────────────────────────────
    meta_train = np.column_stack([oof_lgbm, oof_xgb])
    meta_test = np.column_stack([test_lgbm_avg, test_xgb_avg])

    print("\n  Tuning méta-learner LogReg (2 features → peu de risque d'overfit) :")
    meta_results = {}
    for C in [0.001, 0.01, 0.1, 0.5, 1.0]:
        lr = LogisticRegression(C=C, max_iter=2000, random_state=42)
        lr.fit(meta_train, y_full)
        # Accuracy sur les OOF (légèrement optimiste car LR entraîné sur tout l'OOF)
        oof_pred = lr.predict(meta_train)
        meta_results[C] = accuracy_score(y_full, oof_pred)
        print(f"    C={C:.3f}: OOF acc={meta_results[C]:.4f}  "
              f"[w_LGBM={lr.coef_[0][0]:.3f}, w_XGB={lr.coef_[0][1]:.3f}]")

    # On utilise C=0.1 : assez régularisé, robuste
    lr_final = LogisticRegression(C=0.1, max_iter=2000, random_state=42)
    lr_final.fit(meta_train, y_full)
    stacking_proba = lr_final.predict_proba(meta_test)[:, 1]

    print(f"\n  Méta-learner final (C=0.1) :")
    print(f"    w_LGBM={lr_final.coef_[0][0]:.4f}  w_XGB={lr_final.coef_[0][1]:.4f}  intercept={lr_final.intercept_[0]:.4f}")

    # Baselines comparaison
    avg_proba = (test_lgbm_avg + test_xgb_avg) / 2.0
    oof_avg_acc = accuracy_score(y_full, (meta_train.mean(axis=1) > 0.5).astype(int))
    oof_stack_acc = accuracy_score(y_full, lr_final.predict(meta_train))
    print(f"\n  Comparaison (sur OOF) :")
    print(f"    LGBM seul   : {acc_oof_lgbm:.4f}")
    print(f"    XGB seul    : {acc_oof_xgb:.4f}")
    print(f"    Moyenne     : {oof_avg_acc:.4f}")
    print(f"    LogReg stack: {oof_stack_acc:.4f}  ← utilisé pour la soumission")

    # ── 7. Submission ─────────────────────────────────────────────────────────
    print("\n[7/7] Génération de la soumission...")

    # Calibration : threshold pour retrouver le ratio positif du train
    target_ratio = float(y_full.mean())
    sorted_probs = np.sort(stacking_proba)[::-1]
    n_positive = int(len(stacking_proba) * target_ratio)
    threshold = float(sorted_probs[n_positive])
    preds = (stacking_proba > threshold).astype(int)

    print(f"  Ratio cible (train) : {target_ratio:.4f}")
    print(f"  Threshold calibré   : {threshold:.6f}")
    print(f"  Prédictions         : {preds.sum()} pos / {len(preds)} total ({preds.mean()*100:.1f}%)")

    submission = pd.DataFrame({"ROW_ID": X_test_full["ROW_ID"], "score": preds})
    output_file = "submission_xgb_lgbm_stack.csv"
    submission.to_csv(output_file, index=False)
    print(f"\n  Saved → {output_file}")

    # ── Sauvegarde des modèles ────────────────────────────────────────────────
    print("\n[+] Sauvegarde des modèles...")
    os.makedirs("models", exist_ok=True)

    # Réentraîner sur TOUT le train avec les best params (pas juste les folds)
    lgbm_final = lgb.LGBMClassifier(**best_lgbm_p, random_state=42, n_jobs=-1)
    lgbm_final.fit(X_feat, y_full)

    xgb_final = xgb.XGBClassifier(**best_xgb_p, random_state=42, n_jobs=-1)
    xgb_final.fit(X_feat, y_full)

    joblib.dump(lgbm_final, "models/lgbm_best.pkl")
    joblib.dump(xgb_final,  "models/xgb_best.pkl")
    joblib.dump(lr_final,   "models/meta_lr.pkl")

    with open("models/feature_cols.json", "w") as f:
        json.dump(list(X_feat.columns), f, indent=2)
    with open("models/best_params.json", "w") as f:
        json.dump({"lgbm": best_lgbm_p, "xgb": best_xgb_p}, f, indent=2)

    for fname in ["lgbm_best.pkl", "xgb_best.pkl", "meta_lr.pkl",
                  "feature_cols.json", "best_params.json"]:
        size_kb = os.path.getsize(f"models/{fname}") / 1024
        print(f"  ✓ models/{fname:<22}  {size_kb:>7.1f} KB")

    # ── Résumé final ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  RÉSUMÉ FINAL")
    print("=" * 70)
    print(f"  LGBM (5-fold purged)  : val={val_lgbm:.4f} ± {std_lgbm:.4f} | gap={gap_lgbm:.4f}")
    print(f"  XGB  (5-fold purged)  : val={val_xgb:.4f} ± {std_xgb:.4f} | gap={gap_xgb:.4f}")
    print(f"  Stacking OOF (LGBM+XGB LogReg C=0.1) : {oof_stack_acc:.4f}")
    print(f"  Corrélation OOF       : {oof_corr:.4f}")
    print(f"  Modèles sauvegardés   : models/lgbm_best.pkl | models/xgb_best.pkl")
    print("=" * 70)


if __name__ == "__main__":
    run()
