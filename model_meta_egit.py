"""
Meta Ensemble Model Eğitimi
============================
3 uzman modelin (UNSW-NB15, CIC-IDS, NSL-KDD) XGBoost çıktılarını
birleştiren bir LogisticRegression meta-model eğitir.

MİMARİ:
  Gelen veri
      ├─► UNSW-NB15 XGBoost → olasılık_1
      ├─► CIC-IDS   XGBoost → olasılık_2
      └─► NSL-KDD   XGBoost → olasılık_3
                                    │
                             [3 olasılık]
                                    │
                           Meta LogisticRegression
                                    │
                            Final Karar (0/1)

STACKING MANTIĞI:
  Her modelin kendi test seti üzerindeki olasılık tahminleri
  meta-modele eğitim verisi olarak verilir.
  Böylece meta-model "hangi uzman ne zaman yanılır" öğrenir.

KULLANIM:
  python model_meta_egit.py

  Gerekli klasörler (aynı dizinde):
    processed_data/      → UNSW-NB15 işlenmiş veri
    cicids_processed/    → CIC-IDS işlenmiş veri
    nslkdd_processed/    → NSL-KDD işlenmiş veri
    cicids_sonuclar/     → CIC-IDS model_xgboost.joblib
    nslkdd_sonuclar/     → NSL-KDD model_xgboost.joblib
    model_sonuclari/     → UNSW-NB15 model_xgboost.joblib
       VEYA ids_model.pkl (orijinal UNSW modeli)

ÇIKTI:
  meta_ensemble/
    meta_model.joblib          → meta LogisticRegression
    unsw_xgb.joblib            → UNSW XGBoost (kopyalanır)
    cicids_xgb.joblib          → CIC-IDS XGBoost (kopyalanır)
    nslkdd_xgb.joblib          → NSL-KDD XGBoost (kopyalanır)
    unsw_features.json         → UNSW özellik listesi
    cicids_features.json       → CIC-IDS özellik listesi
    nslkdd_features.json       → NSL-KDD özellik listesi
    meta_sonuclar.json         → meta model metrikleri
"""

import json
import os
import sys
import shutil

import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
    precision_score, recall_score,
)

# ------------------------------------------------------------------
# AYARLAR
# ------------------------------------------------------------------
OUTPUT_DIR = "meta_ensemble"

# Uzman model yolları (önce dener, bulamazsa alternatife geçer)
UNSW_MODEL_PATHS = [
    os.path.join("model_sonuclari", "model_xgboost.joblib"),  # evrensel script çıktısı
    "ids_model.pkl",                                           # orijinal model
]
CICIDS_MODEL_PATH  = os.path.join("cicids_sonuclar",  "model_xgboost.joblib")

# Veri klasörleri
UNSW_DATA_DIR   = "unsw_processed"   # veri_hazirlama_unsw.py çıktısı
CICIDS_DATA_DIR = "cicids_processed"

RANDOM_STATE = 42


# ------------------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ------------------------------------------------------------------

def yukle_model(paths):
    """Birden fazla olası yoldan ilk bulunanı yükler."""
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        if os.path.exists(p):
            model = joblib.load(p)
            print(f"  Model yüklendi: {p}")
            # ids_model.pkl pipeline içinde → sklearn pipeline
            # Eğer dict ise (eski format): model["model"] al
            if isinstance(model, dict):
                model = model.get("model", model)
            return model, p
    raise FileNotFoundError(f"Model bulunamadı. Denenen yollar: {paths}")


def yukle_veri(data_dir):
    """İşlenmiş X_test ve y_test yükler."""
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).iloc[:, 0]
    return X_test, y_test


def yukle_ozellikler(data_dir):
    """feature_names.json varsa yükler, yoksa X_test sütunlarından alır."""
    path = os.path.join(data_dir, "feature_names.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    X = pd.read_csv(os.path.join(data_dir, "X_test.csv"), nrows=0)
    return X.columns.tolist()


def olasilik_al(model, X):
    """Modelden saldırı olasılığı (sınıf 1) alır."""
    # sklearn Pipeline veya düz model
    try:
        proba = model.predict_proba(X)[:, 1]
    except Exception:
        # predict_proba yoksa predict kullan
        proba = model.predict(X).astype(float)
    return proba


def model_X_hazirla(model, X_test, feature_names):
    """
    Modelin beklediği sütunları X_test'ten seçer.
    UNSW ids_model.pkl Pipeline ise direkt X'i geçer (Pipeline kendi encode eder).
    """
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        # Pipeline kendi feature selection yapıyor, tüm sütunları ver
        return X_test
    # Düz model: sadece eğitimde gördüğü sütunları ver
    try:
        expected = model.feature_names_in_
        mevcut   = [c for c in expected if c in X_test.columns]
        eksik    = [c for c in expected if c not in X_test.columns]
        X = X_test[mevcut].copy()
        for col in eksik:
            X[col] = 0
        return X[list(expected)]
    except AttributeError:
        # feature_names_in_ yoksa (eski sklearn), feature_names kullan
        mevcut = [c for c in feature_names if c in X_test.columns]
        eksik  = [c for c in feature_names if c not in X_test.columns]
        X = X_test[mevcut].copy()
        for col in eksik:
            X[col] = 0
        return X[feature_names]


# ------------------------------------------------------------------
# ANA AKIŞ
# ------------------------------------------------------------------
def main():
    print("=" * 65)
    print("META ENSEMBLE MODEL EĞİTİMİ")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- 1) Uzman modelleri yükle ----
    print("\n[1/5] Uzman modeller yükleniyor...")
    unsw_model,   unsw_src   = yukle_model(UNSW_MODEL_PATHS)
    cicids_model, cicids_src = yukle_model(CICIDS_MODEL_PATH)
    nslkdd_model, nslkdd_src = yukle_model(NSLKDD_MODEL_PATH)

    # ---- 2) Test verileri ve olasılıklar ----
    print("\n[2/5] Her veri setinin test seti üzerinde olasılıklar hesaplanıyor...")

    unsw_features  = yukle_ozellikler(UNSW_DATA_DIR)
    cicids_features = yukle_ozellikler(CICIDS_DATA_DIR)
    nslkdd_features = yukle_ozellikler(NSLKDD_DATA_DIR)

    X_unsw,  y_unsw  = yukle_veri(UNSW_DATA_DIR)
    X_cicids, y_cicids = yukle_veri(CICIDS_DATA_DIR)
    X_nslkdd, y_nslkdd = yukle_veri(NSLKDD_DATA_DIR)

    print(f"  UNSW test:   {X_unsw.shape}")
    print(f"  CIC-IDS test: {X_cicids.shape}")
    print(f"  NSL-KDD test: {X_nslkdd.shape}")

    # Her model SADECE kendi test seti üzerinde tahmin yapar
    # (farklı veri setlerinin sütunları uyumsuz olduğu için cross prediction yapılmaz)
    print("  Her model kendi test seti üzerinde tahmin yapıyor...")
    p_unsw   = olasilik_al(unsw_model,   model_X_hazirla(unsw_model,   X_unsw,   unsw_features))
    p_cicids = olasilik_al(cicids_model, model_X_hazirla(cicids_model, X_cicids, cicids_features))
    p_nslkdd = olasilik_al(nslkdd_model, model_X_hazirla(nslkdd_model, X_nslkdd, nslkdd_features))

    print(f"  UNSW olasılık aralığı:   [{p_unsw.min():.3f}, {p_unsw.max():.3f}]")
    print(f"  CIC-IDS olasılık aralığı: [{p_cicids.min():.3f}, {p_cicids.max():.3f}]")
    print(f"  NSL-KDD olasılık aralığı: [{p_nslkdd.min():.3f}, {p_nslkdd.max():.3f}]")

    # ---- 3) Meta eğitim verisi oluştur ----
    # Her veri seti için meta özellik: [kendi_uzman_olasiligi, 0, 0] / [0, kendi_uzman_olasiligi, 0] / [0, 0, kendi_uzman_olasiligi]
    # Böylece meta model "hangi uzman konuşuyor" öğrenir
    print("\n[3/5] Meta model için eğitim verisi oluşturuluyor...")

    n_unsw   = len(p_unsw)
    n_cicids = len(p_cicids)
    n_nslkdd = len(p_nslkdd)

    # [unsw_prob, cicids_prob, nslkdd_prob] — ilgisiz modeller 0.5 (belirsiz)
    meta_X_unsw   = np.column_stack([p_unsw,            np.full(n_unsw,   0.5), np.full(n_unsw,   0.5)])
    meta_X_cicids = np.column_stack([np.full(n_cicids, 0.5), p_cicids,          np.full(n_cicids, 0.5)])
    meta_X_nslkdd = np.column_stack([np.full(n_nslkdd, 0.5), np.full(n_nslkdd, 0.5), p_nslkdd         ])

    meta_X_train = np.vstack([meta_X_unsw, meta_X_cicids, meta_X_nslkdd])
    meta_y_train = pd.concat([y_unsw, y_cicids, y_nslkdd], ignore_index=True).values

    print(f"  Meta eğitim seti: {meta_X_train.shape}")
    print(f"  Sınıf dağılımı — Normal: {(meta_y_train==0).sum():,}  |  Saldırı: {(meta_y_train==1).sum():,}")

    # ---- 4) Meta model eğit ----
    print("\n[4/5] Meta LogisticRegression eğitiliyor...")
    meta_model = LogisticRegression(
        C=1.0, solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=RANDOM_STATE,
    )
    meta_model.fit(meta_X_train, meta_y_train)

    # Meta model ağırlıkları
    coef = meta_model.coef_[0]
    print(f"\n  Meta model katsayıları:")
    print(f"    UNSW-NB15  ağırlığı: {coef[0]:.4f}")
    print(f"    CIC-IDS    ağırlığı: {coef[1]:.4f}")
    print(f"    NSL-KDD    ağırlığı: {coef[2]:.4f}")

    # ---- 5) Değerlendirme (her test seti üzerinde) ----
    print("\n[5/5] Meta model değerlendiriliyor...")

    sonuclar = {}
    for isim, meta_X, y_true in [
        ("UNSW-NB15",  meta_X_unsw,    y_unsw),
        ("CIC-IDS",    meta_X_cicids,  y_cicids),
        ("NSL-KDD",    meta_X_nslkdd,  y_nslkdd),
    ]:
        y_pred = meta_model.predict(meta_X)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        print(f"\n--- Meta Model | {isim} ---")
        print(classification_report(y_true, y_pred,
              target_names=["Normal (0)", "Saldırı (1)"], zero_division=0))
        print(f"Confusion Matrix:\n{cm}")
        fnr = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
        print(f"False Negative Rate: {fnr:.4f}")
        print(f"False Positive Rate: {fpr:.4f}")

        sonuclar[isim] = {
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "fnr":       float(fnr),
            "fpr":       float(fpr),
        }

    # ---- Modelleri kaydet ----
    print(f"\nModeller '{OUTPUT_DIR}/' klasörüne kaydediliyor...")

    joblib.dump(meta_model, os.path.join(OUTPUT_DIR, "meta_model.joblib"))

    # Uzman modelleri de kopyala (app_u.py tek yerden yüklesin)
    shutil.copy(unsw_src,   os.path.join(OUTPUT_DIR, "unsw_xgb.joblib"))
    shutil.copy(cicids_src, os.path.join(OUTPUT_DIR, "cicids_xgb.joblib"))
    shutil.copy(nslkdd_src, os.path.join(OUTPUT_DIR, "nslkdd_xgb.joblib"))

    # Özellik listelerini kaydet
    with open(os.path.join(OUTPUT_DIR, "unsw_features.json"),   "w") as f:
        json.dump(unsw_features,   f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "cicids_features.json"), "w") as f:
        json.dump(cicids_features, f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "nslkdd_features.json"), "w") as f:
        json.dump(nslkdd_features, f, indent=2)

    with open(os.path.join(OUTPUT_DIR, "meta_sonuclar.json"), "w", encoding="utf-8") as f:
        json.dump({
            "meta_model_katsayilari": {
                "unsw_nb15":  float(coef[0]),
                "cicids":     float(coef[1]),
                "nslkdd":     float(coef[2]),
            },
            "test_sonuclari": sonuclar,
        }, f, indent=2, ensure_ascii=False)

    print(f"  meta_model.joblib    → meta karar verici")
    print(f"  unsw_xgb.joblib      → UNSW-NB15 uzmanı")
    print(f"  cicids_xgb.joblib    → CIC-IDS uzmanı")
    print(f"  nslkdd_xgb.joblib    → NSL-KDD uzmanı")
    print(f"  *_features.json      → her modelin özellik listesi")

    print("\n" + "=" * 65)
    print("TAMAMLANDI")
    print("=" * 65)
    print("\nÖzet:")
    for isim, s in sonuclar.items():
        print(f"  {isim:12s} → F1: {s['f1']:.4f}  FNR: {s['fnr']:.4f}  FPR: {s['fpr']:.4f}")

    print("\nSonraki adım: app_u.py'yi meta ensemble ile güncelle")
    print("  python app_u.py")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"\n[HATA] {e}", file=sys.stderr)
        sys.exit(1)