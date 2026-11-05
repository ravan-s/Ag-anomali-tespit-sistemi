"""
IDS FastAPI servisi
Kaydedilmis modeli (ids_model.pkl) yukler, /tahmin adresinde
gelen ag baglantisi verisini alip "saldiri mi, degil mi" cevabi doner.

Kurulum (bir kere yapilir):
    python -m venv venv
    venv\\Scripts\\activate          (Windows)
    source venv/bin/activate        (Mac/Linux)
    pip install fastapi uvicorn scikit-learn joblib pandas

Calistirma:
    uvicorn app:app --reload

Sonra tarayicida: http://127.0.0.1:8000/docs
adresine gidip otomatik olusan test ekranindan deneyebilirsin.
"""
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

app = FastAPI(title="IDS Tahmin Servisi")

# Model, uygulama baslarken bir kere yuklenir (her istekte degil)
paket = joblib.load("ids_model.pkl")
MODEL = paket["model"]
SUTUNLAR = paket["sutunlar"]
ESIK = paket["esik"]


class Baglanti(BaseModel):
    """
    Bir ag baglantisinin ozellikleri.
    UNSW-NB15'teki tum sutunlari buraya eklemen gerekiyor;
    ornek olarak en onemli birkacini yazdim, gerisini de
    ayni sekilde ekleyeceksin (bir sonraki adimda tamamlayacagiz).
    """
    model_config = ConfigDict(extra="allow")  # simdilik eksik alanlara izin ver

    dur: float
    proto: str
    service: str
    state: str
    sbytes: int
    dbytes: int
    sttl: int
    dttl: int


@app.get("/")
def anasayfa():
    return {"durum": "calisiyor", "esik": ESIK}


@app.post("/tahmin")
def tahmin_et(veri: Baglanti):
    # Gelen veriyi modelin bekledigi sutun sirasina gore bir satirlik tabloya cevir
    satir = pd.DataFrame([veri.model_dump()])
    satir = satir.reindex(columns=SUTUNLAR, fill_value=0)

    olasilik = MODEL.predict_proba(satir)[0, 1]
    saldiri_mi = bool(olasilik >= ESIK)

    return {
        "saldiri_mi": saldiri_mi,
        "olasilik": round(float(olasilik), 4),
        "esik": ESIK,
    }
