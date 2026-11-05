import io
import os
import json
import traceback
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse

app = FastAPI(title="IDS Ensemble Analiz Servisi")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ══════════════════════════════════════════════════════════════════
# MODEL YÜKLEME — Meta Ensemble
# ══════════════════════════════════════════════════════════════════
ENSEMBLE_DIR = "meta_ensemble"
ESIK = 0.5
CICIDS_ESIK = 0.2   # Infiltration gibi gizli saldırıları kaçırmamak için düşük eşik

def _yukle(yol):
    m = joblib.load(yol)
    if isinstance(m, dict):
        m = m.get("model", m)
    return m

try:
    META_MODEL    = _yukle(os.path.join(ENSEMBLE_DIR, "meta_model.joblib"))
    UNSW_MODEL    = _yukle(os.path.join(ENSEMBLE_DIR, "unsw_xgb.joblib"))
    CICIDS_MODEL  = _yukle(os.path.join(ENSEMBLE_DIR, "cicids_xgb.joblib"))
    with open(os.path.join(ENSEMBLE_DIR, "unsw_features.json"))   as f: UNSW_FEATURES   = json.load(f)
    with open(os.path.join(ENSEMBLE_DIR, "cicids_features.json")) as f: CICIDS_FEATURES = json.load(f)

    ENSEMBLE_YUKLENDI = True
    NSLKDD_MODEL  = None
    NSLKDD_FEATURES = []
    print("✓ Meta ensemble modeli yüklendi (UNSW + CIC-IDS + Meta)")
except Exception as e:
    print(f"[UYARI] Ensemble yüklenemedi: {e}")
    print("  Yedek: ids_model.pkl deneniyor...")
    ENSEMBLE_YUKLENDI = False
    UNSW_MODEL = CICIDS_MODEL = NSLKDD_MODEL = META_MODEL = None
    UNSW_FEATURES = CICIDS_FEATURES = []
    NSLKDD_MODEL = None
    NSLKDD_FEATURES = []

# UNSW için orijinal model (daha iyi sonuç veriyor)
UNSW_ORIJINAL_MODEL = None
UNSW_ORIJINAL_ESIK  = 0.5
UNSW_RF_FEATURES_DOSYA = None
for _unsw_yol in ["model_random_forest.joblib", "model_RandomForest.joblib", "model_rf.joblib", "ids_model.pkl"]:
    try:
        paket = joblib.load(_unsw_yol)
        UNSW_ORIJINAL_MODEL = paket.get("model", paket) if isinstance(paket, dict) else paket
        UNSW_ORIJINAL_ESIK  = paket.get("esik", 0.5)   if isinstance(paket, dict) else 0.5
        # Eğer paket dict ise içinden feature listesi al
        if isinstance(paket, dict) and "features" in paket:
            UNSW_RF_FEATURES_DOSYA = paket["features"]
        print(f"✓ UNSW modeli yüklendi: {_unsw_yol}")
        break
    except Exception as e:
        print(f"[UYARI] {_unsw_yol} yüklenemedi: {e}")
else:
    UNSW_ORIJINAL_MODEL = None
    UNSW_ORIJINAL_ESIK  = 0.5

# feature_names.json varsa oku (eğitim scripti bunu üretir)
if UNSW_RF_FEATURES_DOSYA is None:
    for _fn_yol in ["feature_names.json", "unsw_features.json"]:
        try:
            with open(_fn_yol) as _f:
                UNSW_RF_FEATURES_DOSYA = json.load(_f)
            print(f"✓ Özellik listesi yüklendi: {_fn_yol} ({len(UNSW_RF_FEATURES_DOSYA)} özellik)")
            break
        except Exception:
            pass

# Yedek tekil model (ensemble de yoksa)
TEKIL_MODEL = UNSW_ORIJINAL_MODEL

# ── Opsiyonel Jev AI ──────────────────────────────────────────────
JEV_AKTIF    = False
JEV_ESIK_ALT = 0.35
JEV_ESIK_UST = 0.65

try:
    from typesafe_sdk import TypeSafeClient, Noul  # type: ignore
    if os.environ.get("TYPESAFE_API_KEY"):
        JEV_AKTIF = True
        print("Jev AI katmanı AKTIF")
except ImportError:
    pass

def jev_karar(satir: dict):
    try:
        istemci = TypeSafeClient(model="jev-latest")
        yanit = istemci.system_one(
            state=json.dumps(satir, ensure_ascii=False, default=str),
            questions={"saldiri_olasiligi": Noul(
                instructions=(
                    "Bu ağ trafiği kaydı bir siber saldırı mı yoksa normal trafik mi? "
                    "0 = kesinlikle normal, 1 = kesinlikle saldırı olarak "
                    "0-1 arasında bir olasılık döndür."
                )
            )},
        )
        return float(yanit.nouls["saldiri_olasiligi"].noul)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════
# VERİ SETİ ADAPTÖR SİSTEMİ
# ══════════════════════════════════════════════════════════════════

KOLON_HARITASI = {
    "unsw_nb15": {"_native": True},
    "cicids2017": {
        "Flow Duration":               "dur",
        "Protocol":                    "proto",
        "Total Fwd Packets":           "spkts",
        "Total Backward Packets":      "dpkts",
        "Total Length of Fwd Packets": "sbytes",
        "Total Length of Bwd Packets": "dbytes",
        "Fwd TTL Max":                 "sttl",
        "Bwd TTL Max":                 "dttl",
        "Flow Bytes/s":                "rate",
        "Label":                       "label",
    },
    "cicids2018": {
        "Flow Duration":   "dur",
        "Protocol":        "proto",
        "Fwd Pkts/s":      "spkts",
        "Bwd Pkts/s":      "dpkts",
        "TotLen Fwd Pkts": "sbytes",
        "TotLen Bwd Pkts": "dbytes",
        "Fwd TTL Max":     "sttl",
        "Bwd TTL Max":     "dttl",
        "Label":           "label",
    },
}

CIC_ETIKET_MAP = {
    "BENIGN": 0, "Benign": 0, "benign": 0,
    "DDoS": 1, "DoS Hulk": 1, "DoS GoldenEye": 1,
    "DoS slowloris": 1, "DoS Slowhttptest": 1,
    "FTP-Patator": 1, "SSH-Patator": 1,
    "Bot": 1, "PortScan": 1, "Infiltration": 1,
    "Web Attack – Brute Force": 1,
    "Web Attack – XSS": 1,
    "Web Attack – Sql Injection": 1,
    "Web Attack — Brute Force": 1,
    "Web Attack — XSS": 1,
    "Web Attack — Sql Injection": 1,
}

def format_algi(df: pd.DataFrame) -> str:
    kolonlar = set(c.strip().lower() for c in df.columns)
    if "flow duration" in kolonlar or "flow bytes/s" in kolonlar:
        return "cicids2018" if "totlen fwd pkts" in kolonlar else "cicids2017"
    if "sbytes" in kolonlar and "dbytes" in kolonlar and "proto" in kolonlar:
        return "unsw_nb15"
    return "unsw_nb15"

def veriyi_uyarla(df: pd.DataFrame, format_adi: str) -> pd.DataFrame:
    df.columns = df.columns.str.strip()
    if format_adi == "otomatik":
        format_adi = format_algi(df)
    harita = KOLON_HARITASI.get(format_adi, {})
    if not harita.get("_native"):
        yeniden = {k: v for k, v in harita.items()
                   if k != "_native" and v is not None and k in df.columns}
        df = df.rename(columns=yeniden)

    if "label" in df.columns:
        if format_adi.startswith("cicids"):
            df["label"] = df["label"].map(CIC_ETIKET_MAP).fillna(1).astype(int)
        elif format_adi == "kdd99" and df["label"].dtype == object:
            df["label"] = (df["label"].str.lower() != "normal").astype(int)

    sb = df["sbytes"] if "sbytes" in df.columns else pd.Series(0, index=df.index)
    db = df["dbytes"] if "dbytes" in df.columns else pd.Series(0, index=df.index)
    sp = df["spkts"]  if "spkts"  in df.columns else pd.Series(0, index=df.index)
    dp = df["dpkts"]  if "dpkts"  in df.columns else pd.Series(0, index=df.index)

    if "total_bytes"       not in df.columns: df["total_bytes"]       = sb.fillna(0) + db.fillna(0)
    if "byte_ratio"        not in df.columns: df["byte_ratio"]        = sb.fillna(0) / (db.fillna(0) + 1)
    if "total_pkts"        not in df.columns: df["total_pkts"]        = sp.fillna(0) + dp.fillna(0)
    if "pkt_ratio"         not in df.columns: df["pkt_ratio"]         = sp.fillna(0) / (dp.fillna(0) + 1)
    if "bytes_per_pkt_src" not in df.columns: df["bytes_per_pkt_src"] = sb.fillna(0) / (sp.fillna(0) + 1)
    if "bytes_per_pkt_dst" not in df.columns: df["bytes_per_pkt_dst"] = db.fillna(0) / (dp.fillna(0) + 1)
    if "sttl" in df.columns and "dttl" in df.columns:
        if "ttl_diff" not in df.columns:
            df["ttl_diff"] = (df["sttl"].fillna(0) - df["dttl"].fillna(0)).abs()

    return df

# ══════════════════════════════════════════════════════════════════
# ENSEMBLE INFERENCE
# ══════════════════════════════════════════════════════════════════

def _X_hazirla(X: pd.DataFrame, feature_names: list) -> pd.DataFrame:
    """Modelin beklediği sütunları seç, eksikleri 0 ile doldur."""
    from sklearn.pipeline import Pipeline
    mevcut = [c for c in feature_names if c in X.columns]
    eksik  = [c for c in feature_names if c not in X.columns]
    result = X[mevcut].copy()
    for col in eksik:
        result[col] = 0
    return result[feature_names]

def _olasilik(model, X: pd.DataFrame, feature_names: list) -> np.ndarray:
    """Modelden saldırı olasılığı al. Pipeline ise direkt ver."""
    from sklearn.pipeline import Pipeline
    if isinstance(model, Pipeline):
        X_in = X  # Pipeline kendi sütun seçimini yapıyor
    else:
        X_in = _X_hazirla(X, feature_names)

    X_in = X_in.replace([np.inf, -np.inf], np.nan).fillna(0)
    for col in X_in.select_dtypes(include="float64").columns:
        X_in[col] = X_in[col].astype("float32")

    try:
        return model.predict_proba(X_in)[:, 1]
    except Exception:
        return model.predict(X_in).astype(float)

def ensemble_tahmin(X: pd.DataFrame, format_adi: str):
    """
    Meta ensemble tahmini:
    - Hangi veri seti formatıysa o uzman modeli tam olasılık verir
    - Diğer iki model 0.5 (belirsiz) olarak işaretlenir
    - Meta model 3 olasılıktan final kararı verir
    """
    n = len(X)

    if format_adi in ("unsw_nb15", "kdd99"):
        if UNSW_ORIJINAL_MODEL is not None:
            X_in = X.copy()

            # Proto one-hot
            proto_val = X_in["proto"].astype(str).str.lower().str.strip() if "proto" in X_in.columns else pd.Series("other", index=X_in.index)
            for cat in ["arp", "ospf", "other", "tcp", "udp", "unas"]:
                X_in[f"proto_{cat}"] = (proto_val == cat).astype(int)

            # Service one-hot
            svc_val = X_in["service"].astype(str).str.lower().str.strip() if "service" in X_in.columns else pd.Series("-", index=X_in.index)
            for cat in ["-", "dhcp", "dns", "ftp", "ftp-data", "http", "irc", "pop3", "radius", "smtp", "snmp", "ssh", "ssl"]:
                X_in[f"service_{cat}"] = (svc_val == cat).astype(int)

            # State one-hot
            state_val = X_in["state"].astype(str).str.upper().str.strip() if "state" in X_in.columns else pd.Series("no", index=X_in.index)
            for cat in ["CON", "ECO", "FIN", "INT", "PAR", "REQ", "RST", "URN", "no"]:
                X_in[f"state_{cat}"] = (state_val == cat).astype(int)

            # Ham kategorik + türetilmiş + id sütunlarını çıkar
            drop_cols = ["proto", "service", "state", "id",
                         "total_bytes", "byte_ratio", "total_pkts", "pkt_ratio",
                         "bytes_per_pkt_src", "bytes_per_pkt_dst", "ttl_diff"]
            X_in = X_in.drop(columns=[c for c in drop_cols if c in X_in.columns])

            # Modelin beklediği 67 sütuna hizala
            UNSW_RF_FEATURES = UNSW_RF_FEATURES_DOSYA if UNSW_RF_FEATURES_DOSYA else [
                "dur","spkts","dpkts","sbytes","dbytes","rate","sttl","dttl",
                "sload","dload","sloss","dloss","sinpkt","dinpkt","sjit","djit",
                "swin","stcpb","dtcpb","dwin","tcprtt","synack","ackdat",
                "smean","dmean","trans_depth","response_body_len",
                "ct_srv_src","ct_state_ttl","ct_dst_ltm","ct_src_dport_ltm",
                "ct_dst_sport_ltm","ct_dst_src_ltm","is_ftp_login","ct_ftp_cmd",
                "ct_flw_http_mthd","ct_src_ltm","ct_srv_dst","is_sm_ips_ports",
                "proto_arp","proto_ospf","proto_other","proto_tcp","proto_udp","proto_unas",
                "service_-","service_dhcp","service_dns","service_ftp","service_ftp-data",
                "service_http","service_irc","service_pop3","service_radius",
                "service_smtp","service_snmp","service_ssh","service_ssl",
                "state_CON","state_ECO","state_FIN","state_INT","state_PAR",
                "state_REQ","state_RST","state_URN","state_no",
            ]
            X_in = _X_hazirla(X_in, UNSW_RF_FEATURES)
            X_in = X_in.replace([np.inf, -np.inf], np.nan).fillna(0)
            for col in X_in.select_dtypes(include="float64").columns:
                X_in[col] = X_in[col].astype("float32")

            olasiliklar = UNSW_ORIJINAL_MODEL.predict_proba(X_in)[:, 1]
            tahminler = (olasiliklar >= UNSW_ORIJINAL_ESIK).astype(int)
            return tahminler, olasiliklar, "model_random_forest.joblib (UNSW uzmanı)"
        p_unsw   = _olasilik(UNSW_MODEL,   X, UNSW_FEATURES)
        p_cicids = np.full(n, 0.5)
        p_nslkdd = np.full(n, 0.5)
        aktif_uzman = "UNSW-NB15"
    elif format_adi in ("cicids2017", "cicids2018"):
        p_unsw   = np.full(n, 0.5)
        p_cicids = _olasilik(CICIDS_MODEL, X, CICIDS_FEATURES)
        p_nslkdd = np.full(n, 0.5)
        aktif_uzman = "CIC-IDS"
        # CIC-IDS için meta yerine direkt düşük eşikle karar ver
        tahminler = (p_cicids >= CICIDS_ESIK).astype(int)
        return tahminler, p_cicids, aktif_uzman
    else:
        # Bilinmeyen format → UNSW ile dene
        p_unsw   = _olasilik(UNSW_MODEL,   X, UNSW_FEATURES)
        p_cicids = np.full(n, 0.5)
        p_nslkdd = np.full(n, 0.5)
        aktif_uzman = "UNSW-NB15 (varsayılan)"

    meta_X = np.column_stack([p_unsw, p_cicids, p_nslkdd])
    olasiliklar = META_MODEL.predict_proba(meta_X)[:, 1]
    tahminler   = (olasiliklar >= ESIK).astype(int)
    return tahminler, olasiliklar, aktif_uzman

# ══════════════════════════════════════════════════════════════════
# CHUNK İŞLEME
# ══════════════════════════════════════════════════════════════════

CHUNK_BOYUTU = 10_000

def on_isle_chunk(df_chunk: pd.DataFrame, format_adi: str):
    if "label" in df_chunk.columns:
        gercek_etiket = np.array([
            0 if str(v).strip().lower() in ("0","normal","benign") else 1
            for v in df_chunk["label"].tolist()
        ], dtype=int)
    else:
        gercek_etiket = None
    saldiri_turu  = df_chunk["attack_cat"].values  if "attack_cat" in df_chunk.columns else None

    drop_cols = [c for c in ["label", "attack_cat"] if c in df_chunk.columns]
    X = df_chunk.drop(columns=drop_cols)

    if ENSEMBLE_YUKLENDI or UNSW_ORIJINAL_MODEL is not None:
        tahminler, olasiliklar, aktif_uzman = ensemble_tahmin(X, format_adi)
    else:
        raise ValueError("Hiçbir model yüklenemedi! model_random_forest.joblib dosyasını kontrol edin.")

    return tahminler, olasiliklar, gercek_etiket, saldiri_turu, aktif_uzman

# ══════════════════════════════════════════════════════════════════
# ENDPOINT'LER
# ══════════════════════════════════════════════════════════════════

@app.post("/tam-analiz")
async def tam_analiz(
    dosya:  UploadFile = File(...),
    format: str        = Form("otomatik"),
):
    if not ENSEMBLE_YUKLENDI and TEKIL_MODEL is None:
        raise HTTPException(status_code=500, detail="Hiçbir model yüklü değil!")

    try:
        contents = await dosya.read()

        df_ornek = pd.read_csv(io.BytesIO(contents), nrows=5)
        df_ornek.columns = df_ornek.columns.str.strip()
        algilanan = format_algi(df_ornek) if format == "otomatik" else format

        tum_tahminler   = []
        tum_olasiliklar = []
        tum_etiketler   = []
        tum_turler      = []
        tum_satirlar    = []
        aktif_uzman_adi = ""

        for chunk in pd.read_csv(io.BytesIO(contents), chunksize=CHUNK_BOYUTU):
            chunk.columns = chunk.columns.str.strip()
            chunk = veriyi_uyarla(chunk, algilanan)

            tahminler, olasiliklar, etiketler, turler, aktif_uzman_adi = on_isle_chunk(chunk, algilanan)

            for idx, (_, row) in enumerate(chunk.iterrows()):
                tur_val = turler[idx] if turler is not None and idx < len(turler) else "-"
                label_raw = row.get("label", None)
                label_int = None
                if label_raw is not None and pd.notna(label_raw):
                    label_int = 0 if str(label_raw).strip().lower() in ("0","normal","benign") else 1
                tum_satirlar.append({
                    "dur":        float(row.get("dur", 0) or 0),
                    "proto":      str(row.get("proto", row.get("Protocol", "-"))),
                    "service":    str(row.get("service", row.get("Service", "-"))),
                    "sbytes":     int(float(row.get("sbytes", row.get("Total Length of Fwd Packets", 0)) or 0)),
                    "dbytes":     int(float(row.get("dbytes", row.get("Total Length of Bwd Packets", 0)) or 0)),
                    "saldiri_turu": str(tur_val) if tur_val and str(tur_val) != "nan" else "-",
                    "olasilik":   round(float(olasiliklar[idx]) * 100, 1),
                    "label":      label_int,
                })

            tum_tahminler.extend(tahminler.tolist())
            tum_olasiliklar.extend(olasiliklar.tolist())
            if etiketler is not None:
                tum_etiketler.extend(etiketler.tolist())
            if turler is not None:
                tum_turler.extend(turler.tolist())

        tahminler_arr = np.array(tum_tahminler)
        toplam_satir  = len(tahminler_arr)

        guvenli_listesi = []
        tehdit_listesi  = []
        for i, (kayit, tahmin) in enumerate(zip(tum_satirlar, tum_tahminler)):
            kayit["satir_no"]      = i + 1
            kayit["gercek_etiket"] = kayit.pop("label")
            kayit["tehdit_mi"]     = tahmin
            (tehdit_listesi if tahmin == 1 else guvenli_listesi).append(kayit)

        dogruluk = kacirilan_orani = yanlis_alarm_orani = None
        if tum_etiketler:
            y  = np.array(tum_etiketler).astype(int)
            dogruluk = round(float((tahminler_arr == y).mean()) * 100, 2)
            sm = y == 1
            nm = y == 0
            kacirilan_orani    = round(float((tahminler_arr[sm] == 0).mean()) * 100, 2) if sm.sum() > 0 else 0.0
            yanlis_alarm_orani = round(float((tahminler_arr[nm] == 1).mean()) * 100, 2) if nm.sum() > 0 else 0.0

        kategori_dagilimi = []
        if tum_turler:
            df_kat = pd.DataFrame({
                "tur":    tum_turler,
                "tahmin": tum_tahminler,
                "gercek": tum_etiketler if tum_etiketler else tum_tahminler,
            })
            for tur, grup in df_kat.groupby("tur"):
                if tum_etiketler:
                    tespit = float((grup["tahmin"] == grup["gercek"]).mean())
                else:
                    tespit = float((grup["tahmin"] == 1).mean())
                kategori_dagilimi.append({
                    "tur":          str(tur),
                    "tespit_orani": round(tespit * 100, 2),
                    "adet":         int(len(grup)),
                })
            kategori_dagilimi.sort(key=lambda x: -x["adet"])

        return JSONResponse(content={
            "toplam_satir":            toplam_satir,
            "guvenli_sayisi":          len(guvenli_listesi),
            "tehdit_sayisi":           len(tehdit_listesi),
            "tehdit_orani":            round(len(tehdit_listesi) / toplam_satir * 100, 2),
            "dogruluk":                dogruluk,
            "kacirilan_orani":         kacirilan_orani,
            "yanlis_alarm_orani":      yanlis_alarm_orani,
            "kategori_dagilimi":       kategori_dagilimi,
            "guvenli_tablosu":         guvenli_listesi[:100],
            "tehdit_tablosu":          tehdit_listesi[:100],
            "algilanan_format":        algilanan,
            "aktif_uzman":             aktif_uzman_adi,
            "ensemble_aktif":          ENSEMBLE_YUKLENDI,
            "jev_devreye_giren_satir": 0,
            "jev_karari_degistirdi":   0,
        })

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Hata: {str(e)}")


@app.get("/durum")
def durum():
    return {
        "ensemble_yuklendi":     ENSEMBLE_YUKLENDI,
        "esik":                  ESIK,
        "jev_aktif":             JEV_AKTIF,
        "desteklenen_formatlar": ["unsw_nb15", "cicids2017", "cicids2018"],
        "uzman_modeller": {
            "unsw_nb15":  UNSW_MODEL  is not None,
            "cicids":     CICIDS_MODEL is not None,
            "meta":       META_MODEL   is not None,
        }
    }


@app.get("/", response_class=HTMLResponse)
def anasayfa():
    with open("ids_demo_u.html", encoding="utf-8") as f:
        return f.read()