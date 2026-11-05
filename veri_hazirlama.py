"""
UNSW-NB15 Veri Temizleme ve Hazirlama Scripti
================================================
Bu script, IDS projesi Asama 2 (Veri Temizleme) adimlarini gerceklestirir:

  1) Eksik / sonsuz (inf) deger kontrolu ve temizligi
  2) Tekrar eden satirlarin kaldirilmasi
  3) Veri sizintisi yaratan sutunlarin (id, attack_cat) ozelliklerden (X) cikarilmasi
  4) [YENI] Asiri uc degerlerin (outlier) winsorization ile sinirlanmasi
  5) [YENI] Carpik (skewed) sayisal ozelliklere log1p donusumu uygulanmasi
  6) Nadir gorulen 'proto' kategorilerinin 'other' altinda gruplanmasi
     (133 farkli protokol var, ~%92'si sadece ilk 5-6 kategoride toplaniyor)
  7) Kategorik sutunlarin (proto, service, state) One-Hot Encoding ile
     sayisallastirilmasi -- SADECE train verisine fit edilir, boylece
     train/test arasinda sutun uyumsuzlugu veya veri sizintisi olmaz
  8) Sinif dengesizligi icin class_weight / scale_pos_weight hesaplanmasi
     (bu veri setinde saldiri orani %68, normal %32 -- gercek dunyanin tersi)
  9) Logistic Regression gibi olcek-duyarli modeller icin ayrica
     StandardScaler uygulanmis bir versiyon uretilmesi
     (Random Forest / XGBoost olcekten etkilenmedigi icin olceksiz
     versiyonu kullanmaya devam edebilirler)
  10) Tum ciktilarin, bir sonraki asama (Asama 3 - model egitimi) icin
     diske kaydedilmesi

YENI EKLENEN ADIMLAR HAKKINDA (4 ve 5) -- NEDEN EKLENDI
---------------------------------------------------------
Model egitimi asamasinda (model_egitimi.py) RandomForest, LogisticRegression
ve XGBoost ile literatur seviyesinde sonuclar elde edildi, ancak XGBoost
literaturdeki en iyi sonuclarin (F1~0.94-0.95) gerisinde kaldi (bkz.
Rehman et al. 2025, arXiv:2510.17883 -- ayni resmi train/test split'inde
XGBoost F1=0.9465 raporlamis). Hyperparameter aramasi (Optuna) uzerinde
COK sayida deneme yapildi ve hicbiri bu farki kapatamadi -- bu da sorunun
model ayarlarinda degil, VERIYE VERILEN OZELLIKLERDE olabilecegini
gosteriyor.

Bu yuzden iki literatur-destekli on-isleme adimi eklendi:

  a) OUTLIER WINSORIZATION: Ag trafigi verilerinde (bytes, rate, load gibi
     sutunlar) birkac asiri buyuk deger, dagilimi domine edebilir. Yontem
     Rehman et al. degil, baska bir UNSW-NB15 calismasindan (arXiv:2509.15555,
     "Hybrid Deep Learning-Federated Learning Powered IDS...") alinmistir:
     "for a numeric feature x, if max(x) > 10*median(x) and max(x) > 10,
     values above the 95th percentile are capped at that percentile."
     Sinir SADECE train verisinden ogrenilir, ayni sinir teste uygulanir
     (veri sizintisi olmaz).

  b) LOG1P DONUSUMU: Sayisal ozelliklerin CARPIKLIGI (skewness) train
     verisi uzerinde olculur (scipy.stats.skew). Skewness > 1 olan VE
     tum degerleri negatif olmayan sutunlara log1p(x) = log(1+x)
     uygulanir. Bu, istatistikte carpik/heavy-tailed veri icin standart,
     yaygin kullanilan bir tekniktir (bkz. "Applied Predictive Modeling",
     Kuhn & Johnson 2013). Hangi sutunlarin donusturulecegi SADECE train
     verisinden, otomatik olarak belirlenir (hardcoded bir liste degil) --
     boylece sonuc keyfi degil, veriye dayali ve tekrar uretilebilir olur.

  Bu iki adim GARANTI bir iyilesme saglamaz -- ama model_egitimi.py'de
  daha once denenen (hyperparameter genisletme, threshold optimizasyonu,
  ornek agirliklandirma) yontemlerden farkli olarak, DOGRUDAN LITERATURDE
  BU VERI SETI ICIN kullanilmis, kanitlanmis teknikledir.

KULLANIM
--------
  1) Asagidaki "AYARLAR" bolumundeki TRAIN_PATH / TEST_PATH degiskenlerini
     kendi dosya yollariniza gore guncelleyin (dosyalar script ile ayni
     klasordeyse degistirmenize gerek yok).
  2) Gerekli kutuphaneler yuklu degilse:
         pip install pandas numpy scikit-learn scipy
  3) Calistirin:
         python veri_hazirlama.py

CIKTI
-----
  processed_data/ klasoru altinda:
    X_train.csv, X_test.csv                -> RF / XGBoost icin (olceksiz)
    X_train_scaled.csv, X_test_scaled.csv  -> Logistic Regression icin (olcekli)
    y_train.csv, y_test.csv                -> etiketler (0=normal, 1=saldiri)
    attack_cat_train.csv, attack_cat_test.csv -> Asama 7 hata analizi icin saklanir
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------------
# AYARLAR -- kendi dosya yollariniza gore guncelleyin
# ------------------------------------------------------------------
TRAIN_PATH = "UNSW_NB15_training-set.csv"
TEST_PATH = "UNSW_NB15_testing-set.csv"
OUTPUT_DIR = "processed_data"

# Train setinde bu orandan (%1) daha az gorulen 'proto' kategorileri
# 'other' altinda toplanir. Kategori tanimi SADECE train'den ogrenilir.
PROTO_MIN_FREQUENCY = 0.01

# [YENI] Outlier winsorization ayarlari (arXiv:2509.15555'ten)
WINSORIZE_PERCENTILE = 95

# [YENI] Log1p donusumu icin carpiklik (skewness) esigi
SKEW_THRESHOLD = 1.0

CATEGORICAL_COLS = ["proto", "service", "state"]
LEAKAGE_COLS = ["id", "attack_cat", "label"]  # X'e asla girmeyecek sutunlar


# ------------------------------------------------------------------
# YARDIMCI FONKSIYONLAR
# ------------------------------------------------------------------
def load_data(train_path, test_path):
    """CSV dosyalarini yukler; dosya bulunamazsa anlasilir hata verir."""
    if not os.path.exists(train_path):
        raise FileNotFoundError(
            f"Train dosyasi bulunamadi: '{train_path}'. "
            f"Script'in basindaki TRAIN_PATH degiskenini kontrol edin."
        )
    if not os.path.exists(test_path):
        raise FileNotFoundError(
            f"Test dosyasi bulunamadi: '{test_path}'. "
            f"Script'in basindaki TEST_PATH degiskenini kontrol edin."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Beklenen sutunlarin var oldugunu dogrula (erken ve anlasilir hata icin)
    required_cols = set(CATEGORICAL_COLS) | set(LEAKAGE_COLS)
    for name, df in [("train", train), ("test", test)]:
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"{name} dosyasinda beklenen sutunlar eksik: {missing}")

    return train, test


def basic_clean(df, name="veri"):
    """
    Eksik deger, sonsuz (inf) deger ve tekrar eden satirlari temizler.
    Bu veri setinde bu sorunlarin olmadigi onceden dogrulandi, ancak
    script'in farkli bir veri versiyonuyla da guvenli calismasi icin
    kontroller yine de yapilir.
    """
    df = df.copy()
    before = len(df)

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    n_missing = int(df.isnull().sum().sum())
    if n_missing > 0:
        print(f"  [{name}] {n_missing} eksik/sonsuz deger bulundu -> satirlar cikariliyor.")
        df = df.dropna().reset_index(drop=True)
    else:
        print(f"  [{name}] Eksik/sonsuz deger yok.")

    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        print(f"  [{name}] {n_dup} tekrar eden satir bulundu -> cikariliyor.")
        df = df.drop_duplicates().reset_index(drop=True)
    else:
        print(f"  [{name}] Tekrar eden satir yok.")

    after = len(df)
    if before != after:
        print(f"  [{name}] {before} -> {after} satir (temizlik sonrasi).")

    return df


def winsorize_outliers(X_train, X_test, numeric_cols, percentile=95):
    """
    [YENI] Asiri uc degerleri sinirlar (arXiv:2509.15555 yontemi).
    Bir sutun icin max(x) > 10*medyan(x) VE max(x) > 10 ise, o sutunun
    train'deki [percentile]. persentili ustundeki degerler bu sinira
    cekilir (capping). Sinir SADECE train'den ogrenilir, ayni sinir
    teste de uygulanir -- veri sizintisi olmaz.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    capped_cols = []

    for col in numeric_cols:
        train_max = X_train[col].max()
        train_median = X_train[col].median()
        if train_median > 0 and train_max > 10 * train_median and train_max > 10:
            cap_value = np.percentile(X_train[col], percentile)
            X_train[col] = X_train[col].clip(upper=cap_value)
            X_test[col] = X_test[col].clip(upper=cap_value)
            capped_cols.append(col)

    print(f"  Winsorization uygulanan sutun sayisi: {len(capped_cols)} / {len(numeric_cols)}")
    if capped_cols:
        print(f"  Ornek sutunlar: {capped_cols[:8]}{'...' if len(capped_cols) > 8 else ''}")
    return X_train, X_test


def log_transform_skewed(X_train, X_test, numeric_cols, skew_threshold=1.0):
    """
    [YENI] Carpik (right-skewed) sayisal sutunlara log1p(x) = log(1+x)
    uygular. Hangi sutunlarin donusturulecegi SADECE train verisinin
    carpiklik (skewness) degerine gore, otomatik olarak belirlenir --
    keyfi/hardcoded bir liste degildir. log1p, negatif olmayan degerler
    icin guvenlidir (bu veri setindeki sayisal ozellikler -- byte/paket
    sayilari, oranlar, sureler -- dogasi geregi negatif olamaz).
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    transformed_cols = []

    for col in numeric_cols:
        if (X_train[col] < 0).any():
            continue  # log1p negatif degerlerde guvenli degil, atla
        col_skew = skew(X_train[col])
        if col_skew > skew_threshold:
            X_train[col] = np.log1p(X_train[col])
            X_test[col] = np.log1p(X_test[col].clip(lower=0))  # test'te olasi negatifi guvenli kes
            transformed_cols.append((col, round(col_skew, 2)))

    print(f"  Log1p uygulanan sutun sayisi: {len(transformed_cols)} / {len(numeric_cols)}")
    if transformed_cols:
        preview = ", ".join(f"{c} (skew={s})" for c, s in transformed_cols[:8])
        print(f"  Ornekler: {preview}{'...' if len(transformed_cols) > 8 else ''}")
    return X_train, X_test


def group_rare_categories(train_series, test_series, min_frequency):
    """
    Train setindeki frekansi min_frequency'nin altinda kalan kategorileri
    'other' altinda toplar. Kategori listesi SADECE train'den ogrenilir ve
    ayni sekilde test setine uygulanir -- boylece veri sizintisi olmaz.
    Test setinde train'de hic gorulmemis bir kategori cikarsa, o da otomatik
    olarak 'other' sayilir.
    """
    freqs = train_series.value_counts(normalize=True)
    keep_categories = freqs[freqs >= min_frequency].index.tolist()

    train_grouped = train_series.where(train_series.isin(keep_categories), "other")
    test_grouped = test_series.where(test_series.isin(keep_categories), "other")

    print(
        f"  '{train_series.name}': {train_series.nunique()} kategori -> "
        f"{len(keep_categories)} korunan + 'other' "
        f"(toplam {train_grouped.nunique()} kategori)"
    )
    return train_grouped, test_grouped


def make_onehot_encoder():
    """
    Farkli scikit-learn versiyonlariyla uyumlu OneHotEncoder olusturur.
    sklearn >= 1.2 'sparse_output', daha eski versiyonlar 'sparse' parametresini kullanir.
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def encode_categoricals(train_df, test_df, categorical_cols):
    """
    Kategorik sutunlari One-Hot Encoding ile sayisallastirir.
    handle_unknown='ignore' sayesinde test setinde train'de hic gorulmemis
    bir kategori cikarsa hata vermez (o satir icin tum ilgili sutunlar 0 olur).
    Encoder SADECE train verisine fit edilir.
    """
    encoder = make_onehot_encoder()
    encoder.fit(train_df[categorical_cols])

    train_encoded = pd.DataFrame(
        encoder.transform(train_df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=train_df.index,
    )
    test_encoded = pd.DataFrame(
        encoder.transform(test_df[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=test_df.index,
    )
    return train_encoded, test_encoded, encoder


# ------------------------------------------------------------------
# ANA AKIS
# ------------------------------------------------------------------
def main():
    print("=" * 70)
    print("UNSW-NB15 VERI TEMIZLEME VE HAZIRLAMA")
    print("=" * 70)

    # 1) Veriyi yukle
    print("\n[1/10] Veri yukleniyor...")
    train_raw, test_raw = load_data(TRAIN_PATH, TEST_PATH)
    print(f"  Train: {train_raw.shape}   Test: {test_raw.shape}")

    # 2) Temel temizlik (eksik / inf / duplicate)
    print("\n[2/10] Temel temizlik yapiliyor...")
    train = basic_clean(train_raw, "train")
    test = basic_clean(test_raw, "test")

    # 3) Etiket ve saldiri turunu ayir (veri sizintisini onlemek icin)
    print("\n[3/10] Etiketler ayriliyor...")
    y_train = train["label"].astype(int).reset_index(drop=True)
    y_test = test["label"].astype(int).reset_index(drop=True)
    attack_cat_train = train["attack_cat"].reset_index(drop=True)
    attack_cat_test = test["attack_cat"].reset_index(drop=True)
    print(f"  y_train dagilimi: {y_train.value_counts().to_dict()}")
    print(f"  y_test  dagilimi: {y_test.value_counts().to_dict()}")

    # 4) Ozellik (X) setlerini hazirla: id, attack_cat, label CIKAR
    print("\n[4/10] Sizinti yaratan sutunlar (id, attack_cat, label) cikariliyor...")
    X_train_raw = train.drop(columns=LEAKAGE_COLS).reset_index(drop=True)
    X_test_raw = test.drop(columns=LEAKAGE_COLS).reset_index(drop=True)

    numeric_cols_raw = [
        c for c in X_train_raw.columns
        if c not in CATEGORICAL_COLS and pd.api.types.is_numeric_dtype(X_train_raw[c])
    ]

    # 5) [YENI] Outlier winsorization (arXiv:2509.15555 yontemi)
    print("\n[5/10] Asiri uc degerler (outlier) siniriyor (winsorization)...")
    X_train_raw, X_test_raw = winsorize_outliers(
        X_train_raw, X_test_raw, numeric_cols_raw, percentile=WINSORIZE_PERCENTILE
    )

    # 6) [YENI] Carpik sayisal ozelliklere log1p donusumu
    print("\n[6/10] Carpik (skewed) ozelliklere log1p donusumu uygulaniyor...")
    X_train_raw, X_test_raw = log_transform_skewed(
        X_train_raw, X_test_raw, numeric_cols_raw, skew_threshold=SKEW_THRESHOLD
    )

    # 7) 'proto' icin nadir kategori gruplama
    #    (service: 13, state: 9 kategori zaten makul, gruplamaya gerek yok)
    print("\n[7/10] Nadir kategoriler gruplaniyor...")
    proto_train_grouped, proto_test_grouped = group_rare_categories(
        X_train_raw["proto"], X_test_raw["proto"], PROTO_MIN_FREQUENCY
    )
    X_train_raw["proto"] = proto_train_grouped
    X_test_raw["proto"] = proto_test_grouped

    # 8) Kategorik sutunlari One-Hot Encode et
    print("\n[8/10] One-Hot Encoding uygulaniyor...")
    train_cat_encoded, test_cat_encoded, encoder = encode_categoricals(
        X_train_raw, X_test_raw, CATEGORICAL_COLS
    )
    print(f"  Kategorik sutunlardan uretilen yeni sutun sayisi: {train_cat_encoded.shape[1]}")

    numeric_cols = [c for c in X_train_raw.columns if c not in CATEGORICAL_COLS]
    X_train = pd.concat(
        [X_train_raw[numeric_cols].reset_index(drop=True), train_cat_encoded], axis=1
    )
    X_test = pd.concat(
        [X_test_raw[numeric_cols].reset_index(drop=True), test_cat_encoded], axis=1
    )

    # Guvenlik kontrolu: train/test sutunlari birebir ayni ve ayni sirada olmali
    assert list(X_train.columns) == list(X_test.columns), (
        "HATA: Train ve test sutunlari uyusmuyor! Encoding adiminda bir sorun var."
    )
    assert X_train.isnull().sum().sum() == 0, "HATA: X_train icinde beklenmedik NaN var."
    assert X_test.isnull().sum().sum() == 0, "HATA: X_test icinde beklenmedik NaN var."
    print(
        f"  Nihai ozellik sayisi: {X_train.shape[1]} "
        f"({len(numeric_cols)} sayisal + {train_cat_encoded.shape[1]} one-hot)"
    )

    # 9) Sinif agirliklarini hesapla (dengesizlik icin)
    print("\n[9/10] Sinif dengesizligi degerlendiriliyor...")
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weight_dict = {int(c): float(w) for c, w in zip(classes, weights)}
    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"  Onerilen class_weight (RF / Logistic Regression icin): {class_weight_dict}")
    print(f"  Onerilen scale_pos_weight (XGBoost icin): {scale_pos_weight:.4f}")

    # Logistic Regression icin olceklenmis versiyon
    # (Random Forest / XGBoost olcekten etkilenmez, onlar icin X_train yeterli)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=X_test.columns, index=X_test.index
    )
    print("  Olcekleme tamamlandi (scaler SADECE train'e fit edildi).")

    # 10) Ciktilari diske kaydet
    print("\n[10/10] Sonuclar diske kaydediliyor...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    X_train_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_train_scaled.csv"), index=False)
    X_test_scaled.to_csv(os.path.join(OUTPUT_DIR, "X_test_scaled.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)
    attack_cat_train.to_csv(os.path.join(OUTPUT_DIR, "attack_cat_train.csv"), index=False)
    attack_cat_test.to_csv(os.path.join(OUTPUT_DIR, "attack_cat_test.csv"), index=False)

    print("\n" + "=" * 70)
    print(f"TAMAMLANDI -- islenmis dosyalar '{OUTPUT_DIR}/' klasorune kaydedildi:")
    print("=" * 70)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        path = os.path.join(OUTPUT_DIR, f)
        size_kb = os.path.getsize(path) / 1024
        print(f"  - {f:<28} ({size_kb:>8.1f} KB)")

    print("\nOZET")
    print(f"  X_train / X_train_scaled : {X_train.shape}")
    print(f"  X_test  / X_test_scaled  : {X_test.shape}")
    print(f"  y_train dagilimi         : {y_train.value_counts().to_dict()}")
    print(f"  y_test  dagilimi         : {y_test.value_counts().to_dict()}")
    print("\nSonraki adim (Asama 3): model_egitimi.py'yi tekrar calistir --")
    print("processed_data/ artik winsorize + log-transform edilmis ozellikler iceriyor.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001 - kullaniciya anlasilir hata mesaji vermek icin
        print(f"\n[HATA] Script durduruldu: {exc}", file=sys.stderr)
        sys.exit(1)
