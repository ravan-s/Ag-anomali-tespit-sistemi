"""
UNSW-NB15: Random Forest icin karar esigi (threshold) ayarlama

Colab'da kullanim:
    1) Bu dosyanin tamamini kopyala, yeni bir hucreye yapistir
    2) TRAIN ve TEST satirlarini kendi Drive yolunla degistir
    3) Hucreyi calistir (yaklasik 1 dakika surer, tek model egitiliyor)
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# --- BURAYI KENDI DRIVE YOLUNLA DEGISTIR ---
TRAIN = "/content/drive/MyDrive/IDS_Projesi/UNSW_NB15_training-set.csv"
TEST = "/content/drive/MyDrive/IDS_Projesi/UNSW_NB15_testing-set.csv"
# --------------------------------------------

CATEGORICAL = ["proto", "service", "state"]
DROP = ["id", "attack_cat", "label"]

# Denenecek esik degerleri: 0.30'dan 0.70'e, 0.05 araliklarla
ESIKLER = np.arange(0.30, 0.75, 0.05)


def load(path):
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def main():
    train, test = load(TRAIN), load(TEST)

    X_train = train.drop(columns=[c for c in DROP if c in train.columns])
    y_train = train["label"]
    X_test = test.drop(columns=[c for c in DROP if c in test.columns])
    y_test = test["label"]
    numeric = [c for c in X_train.columns if c not in CATEGORICAL]

    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ("num", "passthrough", numeric),
    ])
    model = Pipeline([
        ("pre", pre),
        ("rf", RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)),
    ])

    print("Model egitiliyor (bir kere egitilip, farkli esiklerle test edilecek)...")
    model.fit(X_train, y_train)

    # predict_proba: her satir icin "saldiri olma olasiligi" (0 ile 1 arasi)
    proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 65)
    print(f"{'Esik':>6} | {'Dogruluk':>9} | {'Kacirilan saldiri':>18} | {'Yanlis alarm':>13}")
    print("=" * 65)

    sonuclar = []
    for esik in ESIKLER:
        pred = (proba >= esik).astype(int)
        tp = ((pred == 1) & (y_test == 1)).sum()
        tn = ((pred == 0) & (y_test == 0)).sum()
        fp = ((pred == 1) & (y_test == 0)).sum()
        fn = ((pred == 0) & (y_test == 1)).sum()

        dogruluk = (tp + tn) / (tp + tn + fp + fn)
        kacirilan = fn / (fn + tp)
        yanlis_alarm = fp / (fp + tn)

        sonuclar.append({"esik": round(esik, 2), "dogruluk": dogruluk,
                          "kacirilan_saldiri": kacirilan, "yanlis_alarm": yanlis_alarm})
        print(f"{esik:>6.2f} | {dogruluk:>9.3f} | {kacirilan:>18.3f} | {yanlis_alarm:>13.3f}")

    print("\nNot: Esik yukseldikce -> model daha 'temkinli' olur,")
    print("     yanlis alarm azalir ama kacirilan saldiri artabilir. Tam tersi de gecerli.")

    ozet = pd.DataFrame(sonuclar)
    en_dengeli = ozet.assign(
        skor=lambda d: d["kacirilan_saldiri"] + d["yanlis_alarm"]
    ).sort_values("skor").iloc[0]
    print(f"\nİki hatayi toplamda en aza indiren esik: {en_dengeli['esik']}")
    print("(Bu sadece bir oneri, IDS'te kacirilan saldiriya daha fazla agirlik vermek isteyebilirsiniz,")
    print(" o zaman kacirilan_saldiri sutunu dusuk olan bir esigi siz secebilirsiniz.)")


if __name__ == "__main__":
    main()
