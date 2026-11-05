"""
Evrensel Model Eğitim Scripti
==============================
Herhangi bir veri seti için çalışır:
  - UNSW-NB15  → DATA_DIR = "processed_data"
  - CIC-IDS    → DATA_DIR = "cicids_processed"
  - NSL-KDD    → DATA_DIR = "nslkdd_processed"

model_egitimi.py ile birebir aynı mantık (RF + LR + XGBoost + Ensemble),
sadece DATA_DIR ve OUTPUT_DIR komut satırından verilebilir:

  python model_egitimi_evrensel.py --data cicids_processed --out cicids_sonuclar
  python model_egitimi_evrensel.py --data nslkdd_processed --out nslkdd_sonuclar
  python model_egitimi_evrensel.py --data processed_data   --out unsw_sonuclar

Tüm modeller OUTPUT_DIR/model_*.joblib olarak kaydedilir.
Meta ensemble scripti bu dosyaları kullanır.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# VARSAYILAN AYARLAR (komut satırıyla geçersiz kılınabilir)
# ------------------------------------------------------------------
N_TRIALS           = 40
CV_FOLDS           = 3
SEARCH_SAMPLE_SIZE = 50000
RANDOM_STATE       = 42
N_JOBS             = -1


# ------------------------------------------------------------------
# VERİ YÜKLEME
# ------------------------------------------------------------------
def load_processed_data(data_dir):
    required = [
        "X_train.csv", "X_test.csv", "X_train_scaled.csv",
        "X_test_scaled.csv", "y_train.csv", "y_test.csv",
    ]
    missing = [f for f in required if not os.path.exists(os.path.join(data_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"'{data_dir}' klasöründe eksik dosyalar: {missing}\n"
            f"Önce ilgili veri_hazirlama_*.py scriptini çalıştırın."
        )

    X_train        = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test         = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    X_train_scaled = pd.read_csv(os.path.join(data_dir, "X_train_scaled.csv"))
    X_test_scaled  = pd.read_csv(os.path.join(data_dir, "X_test_scaled.csv"))
    y_train        = pd.read_csv(os.path.join(data_dir, "y_train.csv")).iloc[:, 0]
    y_test         = pd.read_csv(os.path.join(data_dir, "y_test.csv")).iloc[:, 0]

    return X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test


def load_attack_categories(data_dir, filename="attack_cat_test.csv"):
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).iloc[:, 0]


def make_search_subset(X, X_scaled, y, sample_size, random_state):
    n = len(y)
    sample_size = min(sample_size, n)
    if sample_size >= n:
        return X.copy(), X_scaled.copy(), y.copy()
    idx_all = np.arange(n)
    idx_search, _ = train_test_split(
        idx_all, train_size=sample_size, stratify=y, random_state=random_state
    )
    return (
        X.iloc[idx_search].reset_index(drop=True),
        X_scaled.iloc[idx_search].reset_index(drop=True),
        y.iloc[idx_search].reset_index(drop=True),
    )


# ------------------------------------------------------------------
# OPTUNA OBJECTIVE FONKSİYONLARI
# ------------------------------------------------------------------
def rf_objective(trial, X, y):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 50, 200),
        max_depth=trial.suggest_int("max_depth", 5, 25),
        min_samples_split=trial.suggest_int("min_samples_split", 2, 15),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
    )
    model = RandomForestClassifier(
        **params, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1).mean()


def lr_objective(trial, X, y):
    params = dict(C=trial.suggest_float("C", 1e-3, 10.0, log=True))
    model = LogisticRegression(
        **params, solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=N_JOBS).mean()


def xgb_objective(trial, X, y, scale_pos_weight):
    params = dict(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        max_depth=trial.suggest_int("max_depth", 3, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        subsample=trial.suggest_float("subsample", 0.7, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.7, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 6),
        reg_alpha=trial.suggest_float("reg_alpha", 0.0, 2.0),
        reg_lambda=trial.suggest_float("reg_lambda", 0.0, 2.0),
    )
    model = XGBClassifier(
        **params, scale_pos_weight=scale_pos_weight, random_state=RANDOM_STATE,
        n_jobs=N_JOBS, eval_metric="logloss", tree_method="hist",
    )
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=1).mean()


# ------------------------------------------------------------------
# DEĞERLENDİRME
# ------------------------------------------------------------------
def evaluate_model(y_test, y_pred, model_name):
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "false_negative_rate": float(fn / (fn + tp)) if (fn + tp) > 0 else float("nan"),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else float("nan"),
        "confusion_matrix": cm.tolist(),
    }
    print(f"\n--- {model_name} | Test Sonuçları ---")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Saldırı (1)"], zero_division=0))
    print(f"Confusion Matrix:\n{cm}")
    print(f"False Negative Rate: {metrics['false_negative_rate']:.4f}")
    print(f"False Positive Rate: {metrics['false_positive_rate']:.4f}")
    return metrics


def save_feature_importance(importances, feature_names, model_name, output_dir, top_n=15):
    if len(importances) == 0:
        return
    idx = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_names[i] for i in idx]
    top_values   = importances[idx]
    plt.figure(figsize=(8, 6))
    plt.barh(top_features[::-1], top_values[::-1])
    plt.title(f"{model_name} - En Önemli {top_n} Özellik")
    plt.xlabel("Önem Skoru")
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"feature_importance_{model_name}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    print(f"Feature importance grafiği: {out_path}")


def evaluate_by_attack_category(y_test, y_pred, attack_cat, model_name, output_dir):
    df = pd.DataFrame({
        "attack_cat": pd.Series(attack_cat).reset_index(drop=True),
        "y_pred": pd.Series(y_pred).reset_index(drop=True),
    })
    rows = []
    for cat, grp in df.groupby("attack_cat"):
        label = "Normal" if str(cat).lower() in ("normal", "benign") else cat
        metric = "false_positive_rate" if str(cat).lower() in ("normal", "benign") else "detection_rate"
        rows.append({
            "category": label,
            "count": len(grp),
            "metric": metric,
            "value": round((grp["y_pred"] == 1).mean(), 4),
        })
    result_df = pd.DataFrame(rows).sort_values("value")
    print(f"\n--- {model_name} | Attack Category Breakdown ---")
    print(result_df.to_string(index=False))
    out_path = os.path.join(output_dir, f"category_breakdown_{model_name}.csv")
    result_df.to_csv(out_path, index=False)
    return result_df


# ------------------------------------------------------------------
# ANA AKIŞ
# ------------------------------------------------------------------
def main(data_dir, output_dir):
    script_start = time.time()
    print("=" * 70)
    print(f"MODEL EĞİTİMİ: {data_dir}")
    print(f"(N_TRIALS={N_TRIALS}, arama örneklemi={SEARCH_SAMPLE_SIZE})")
    print("=" * 70)

    os.makedirs(output_dir, exist_ok=True)

    print(f"\n[1/6] Veriler yükleniyor: {data_dir}/")
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = load_processed_data(data_dir)
    print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")

    attack_cat_test = load_attack_categories(data_dir)
    if attack_cat_test is not None:
        print("  attack_cat_test.csv bulundu — kategori analizi yapılacak.")

    feature_names    = X_train.columns.tolist()
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    print("\n[2/6] Optuna arama örneklemi hazırlanıyor...")
    X_search, X_search_scaled, y_search = make_search_subset(
        X_train, X_train_scaled, y_train, SEARCH_SAMPLE_SIZE, RANDOM_STATE
    )
    print(f"  Arama örneklemi: {X_search.shape}")

    results        = []
    trained_models = {}

    # ---- RANDOM FOREST ----
    print("\n[3/6] Random Forest optimize ediliyor...")
    t0 = time.time()
    study_rf = optuna.create_study(direction="maximize", study_name="rf")
    study_rf.optimize(lambda t: rf_objective(t, X_search, y_search), n_trials=N_TRIALS)
    print(f"  Süre: {time.time()-t0:.1f}s | En iyi params: {study_rf.best_params}")

    best_rf = RandomForestClassifier(
        **study_rf.best_params, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
    )
    t0 = time.time()
    best_rf.fit(X_train, y_train)
    print(f"  Eğitim süresi (tam veri): {time.time()-t0:.1f}s")

    proba_rf = best_rf.predict_proba(X_test)[:, 1]
    y_pred_rf = best_rf.predict(X_test)
    m_rf = evaluate_model(y_test, y_pred_rf, "RandomForest")
    m_rf["best_params"] = study_rf.best_params
    m_rf["cv_f1_search"] = float(study_rf.best_value)
    results.append(m_rf)
    trained_models["random_forest"] = best_rf
    if attack_cat_test is not None:
        evaluate_by_attack_category(y_test, y_pred_rf, attack_cat_test, "RandomForest", output_dir)
    save_feature_importance(best_rf.feature_importances_, feature_names, "RandomForest", output_dir)

    # ---- LOGISTIC REGRESSION ----
    print("\n[4/6] Logistic Regression optimize ediliyor...")
    t0 = time.time()
    study_lr = optuna.create_study(direction="maximize", study_name="lr")
    study_lr.optimize(lambda t: lr_objective(t, X_search_scaled, y_search), n_trials=N_TRIALS)
    print(f"  Süre: {time.time()-t0:.1f}s | En iyi params: {study_lr.best_params}")

    best_lr = LogisticRegression(
        **study_lr.best_params, solver="lbfgs",
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE,
    )
    t0 = time.time()
    best_lr.fit(X_train_scaled, y_train)
    print(f"  Eğitim süresi (tam veri): {time.time()-t0:.1f}s")

    proba_lr = best_lr.predict_proba(X_test_scaled)[:, 1]
    y_pred_lr = best_lr.predict(X_test_scaled)
    m_lr = evaluate_model(y_test, y_pred_lr, "LogisticRegression")
    m_lr["best_params"] = study_lr.best_params
    m_lr["cv_f1_search"] = float(study_lr.best_value)
    results.append(m_lr)
    trained_models["logistic_regression"] = best_lr
    if attack_cat_test is not None:
        evaluate_by_attack_category(y_test, y_pred_lr, attack_cat_test, "LogisticRegression", output_dir)

    # ---- XGBOOST ----
    print("\n[5/6] XGBoost optimize ediliyor...")
    t0 = time.time()
    study_xgb = optuna.create_study(direction="maximize", study_name="xgb")
    study_xgb.optimize(
        lambda t: xgb_objective(t, X_search, y_search, scale_pos_weight),
        n_trials=N_TRIALS,
    )
    print(f"  Süre: {time.time()-t0:.1f}s | En iyi params: {study_xgb.best_params}")

    best_xgb = XGBClassifier(
        **study_xgb.best_params, scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE, n_jobs=N_JOBS,
        eval_metric="logloss", tree_method="hist",
    )
    t0 = time.time()
    best_xgb.fit(X_train, y_train)
    print(f"  Eğitim süresi (tam veri): {time.time()-t0:.1f}s")

    proba_xgb = best_xgb.predict_proba(X_test)[:, 1]
    y_pred_xgb = best_xgb.predict(X_test)
    m_xgb = evaluate_model(y_test, y_pred_xgb, "XGBoost")
    m_xgb["best_params"] = study_xgb.best_params
    m_xgb["cv_f1_search"] = float(study_xgb.best_value)
    results.append(m_xgb)
    trained_models["xgboost"] = best_xgb
    if attack_cat_test is not None:
        evaluate_by_attack_category(y_test, y_pred_xgb, attack_cat_test, "XGBoost", output_dir)
    save_feature_importance(best_xgb.feature_importances_, feature_names, "XGBoost", output_dir)

    # ---- ENSEMBLE (SOFT VOTING) ----
    print("\n[Ekstra] Soft Voting Ensemble değerlendiriliyor...")
    proba_ens = (proba_rf + proba_lr + proba_xgb) / 3.0
    y_pred_ens = (proba_ens >= 0.5).astype(int)
    m_ens = evaluate_model(y_test, y_pred_ens, "Ensemble_SoftVoting")
    m_ens["best_params"] = {"weights": "equal (1/3 each)", "threshold": 0.5}
    m_ens["cv_f1_search"] = float("nan")
    results.append(m_ens)
    if attack_cat_test is not None:
        evaluate_by_attack_category(y_test, y_pred_ens, attack_cat_test, "Ensemble", output_dir)

    # ---- KAYDET ----
    print(f"\n[6/6] Sonuçlar kaydediliyor → '{output_dir}/'")
    comparison_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Accuracy": round(r["accuracy"], 4),
            "Precision": round(r["precision"], 4),
            "Recall": round(r["recall"], 4),
            "F1": round(r["f1"], 4),
            "FN_Rate": round(r["false_negative_rate"], 4),
            "FP_Rate": round(r["false_positive_rate"], 4),
        }
        for r in results
    ])
    comparison_df.to_csv(os.path.join(output_dir, "model_karsilastirma.csv"), index=False)

    with open(os.path.join(output_dir, "detayli_sonuclar.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Feature names da kaydet (meta model için lazım)
    with open(os.path.join(output_dir, "feature_names.json"), "w") as f:
        json.dump(feature_names, f, indent=2)

    for name, model in trained_models.items():
        path = os.path.join(output_dir, f"model_{name}.joblib")
        joblib.dump(model, path)
        print(f"  Kaydedildi: {path}")

    elapsed = time.time() - script_start
    print("\n" + "=" * 70)
    print("TAMAMLANDI")
    print("=" * 70)
    print(comparison_df.to_string(index=False))
    print(f"\nToplam süre: {elapsed:.1f} saniye")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evrensel IDS Model Eğitimi")
    parser.add_argument(
        "--data", default="processed_data",
        help="İşlenmiş veri klasörü (örn: processed_data, cicids_processed, nslkdd_processed)"
    )
    parser.add_argument(
        "--out", default=None,
        help="Sonuç klasörü (belirtilmezse DATA_DIR + '_sonuclar' kullanılır)"
    )
    args = parser.parse_args()
    out = args.out or (args.data.rstrip("/") + "_sonuclar")

    try:
        main(args.data, out)
    except Exception as e:
        print(f"\n[HATA] {e}", file=sys.stderr)
        sys.exit(1)