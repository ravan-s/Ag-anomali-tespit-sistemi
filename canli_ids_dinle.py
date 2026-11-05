"""
canli_ids_dinle.py

Scapy kullanarak canlı ağ trafiğini dinler, paketleri akışlara (Flow) gruplar,
türetilmiş özellikleri hesaplar ve ids_model_genis.pkl modeli ile anlık
saldırı tespiti gerçekleştirir.

Kullanım (Yönetici/Admin yetkisiyle):
    python canli_ids_dinle.py
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
from collections import defaultdict
from scapy.all import conf, sniff, IP, TCP, UDP

try:
    conf.L3socket
except Exception:
    pass

# ══════════════════════════════════════════════════════════════════
# MODEL VE AYARLARIN YÜKLENMESİ
# ══════════════════════════════════════════════════════════════════
MODEL_DOSYASI = "ids_model_genis.pkl"
AKIS_ZAMAN_ASIMI = 3.0  # Saniye cinsinden akış zaman aşımı (Inactivity Timeout)
KARAR_ESIGI = 0.60      # Yanlış alarmları azaltmak için optimize edilen karar eşiği

if not os.path.exists(MODEL_DOSYASI):
    print(f"❌ HATA: {MODEL_DOSYASI} bulunamadı! Lütfen önce modeli eğitin.")
    exit(1)

print("📦 Model yükleniyor...")
paket_data = joblib.load(MODEL_DOSYASI)
model = paket_data["model"]
sutunlar = paket_data["sutunlar"]

print(f"✅ Model başarıyla yüklendi! (Gerekli Özellik Sayısı: {len(sutunlar)})")

# ══════════════════════════════════════════════════════════════════
# AKIŞ (FLOW) YÖNETİM YAPISI
# ══════════════════════════════════════════════════════════════════
# Akış Anahtarı: (src_ip, dst_ip, src_port, dst_port, proto_str)
akislar = {}

class Akis:
    def __init__(self, src_ip, dst_ip, src_port, dst_port, proto):
        self.src_ip = src_ip
        self.dst_ip = dst_ip
        self.src_port = src_port
        self.dst_port = dst_port
        self.proto = proto
        self.start_time = time.time()
        self.last_time = time.time()
        
        self.sbytes = 0
        self.dbytes = 0
        self.spkts = 0
        self.dpkts = 0

    def paket_ekle(self, p_len, yon="fwd"):
        self.last_time = time.time()
        if yon == "fwd":
            self.spkts += 1
            self.sbytes += p_len
        else:
            self.dpkts += 1
            self.dbytes += p_len

    def ozellikleri_hesapla(self):
        dur = max(self.last_time - self.start_time, 0.0001)
        sb, db = self.sbytes, self.dbytes
        sp, dp = self.spkts, self.dpkts

        total_bytes = sb + db
        byte_ratio = sb / (db + 1)
        total_pkts = sp + dp
        pkt_ratio = sp / (dp + 1)
        bytes_per_pkt_src = sb / (sp + 1)
        bytes_per_pkt_dst = db / (dp + 1)

        row = {
            "dur": dur,
            "sbytes": sb,
            "dbytes": db,
            "spkts": sp,
            "dpkts": dp,
            "byte_ratio": byte_ratio,
            "pkt_ratio": pkt_ratio,
            "bytes_per_pkt_src": bytes_per_pkt_src,
            "bytes_per_pkt_dst": bytes_per_pkt_dst,
            "total_bytes": total_bytes,
            "total_pkts": total_pkts,
            "proto": self.proto
        }
        return row

# ══════════════════════════════════════════════════════════════════
# TAHMİN MİMARİSİ
# ══════════════════════════════════════════════════════════════════
def akis_tahmin_et(akis: Akis):
    raw_dict = akis.ozellikleri_hesapla()
    df = pd.DataFrame([raw_dict])
    
    # One-Hot Encoding
    df_enc = pd.get_dummies(df, columns=["proto"])
    
    # Model sütun hizalama
    df_enc = df_enc.reindex(columns=sutunlar, fill_value=0)
    df_enc = df_enc.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Olasılık tahmini
    saldiri_olasilik = model.predict_proba(df_enc)[0][1]
    
    # Sonuç Raporlama
    zaman_str = time.strftime("%H:%M:%S")
    akıs_ozet = f"{akis.src_ip}:{akis.src_port} ➔ {akis.dst_ip}:{akis.dst_port} [{akis.proto.upper()}]"
    
    if saldiri_olasilik >= KARAR_ESIGI:
        print(f"🚨 [{zaman_str}] SALDIRI TESPİT EDİLDİ! Olasılık: %{saldiri_olasilik*100:.1f} | {akıs_ozet} (Paket: {akis.spkts + akis.dpkts}, Bayt: {akis.sbytes + akis.dbytes})")
    else:
        print(f"🟢 [{zaman_str}] NORMAL TRAFİK | Olasılık: %{saldiri_olasilik*100:.1f} | {akıs_ozet}")

# ══════════════════════════════════════════════════════════════════
# SCAPY PAKET YAKALAMA CALLBACK
# ══════════════════════════════════════════════════════════════════
def paket_islem(pkt):
    if not pkt.haslayer(IP):
        return

    src_ip = pkt[IP].src
    dst_ip = pkt[IP].dst
    p_len = len(pkt)
    
    src_port, dst_port = 0, 0
    proto = "other"

    if pkt.haslayer(TCP):
        src_port = pkt[TCP].sport
        dst_port = pkt[TCP].dport
        proto = "tcp"
    elif pkt.haslayer(UDP):
        src_port = pkt[UDP].sport
        dst_port = pkt[UDP].dport
        proto = "udp"

    # Akış Anahtarı Oluştur (İki yönlü eşleşme için)
    fwd_key = (src_ip, dst_ip, src_port, dst_port, proto)
    rev_key = (dst_ip, src_ip, dst_port, src_port, proto)

    now = time.time()

    if fwd_key in akislar:
        akislar[fwd_key].paket_ekle(p_len, yon="fwd")
    elif rev_key in akislar:
        akislar[rev_key].paket_ekle(p_len, yon="bwd")
    else:
        yeni_akis = Akis(src_ip, dst_ip, src_port, dst_port, proto)
        yeni_akis.paket_ekle(p_len, yon="fwd")
        akislar[fwd_key] = yeni_akis

    # Zaman aşımına uğrayan veya süresi dolan akışları değerlendir
    silinecekler = []
    for k, akis_obj in list(akislar.items()):
        if now - akis_obj.last_time >= AKIS_ZAMAN_ASIMI:
            akis_tahmin_et(akis_obj)
            silinecekler.append(k)

    for k in silinecekler:
        del akislar[k]

# ══════════════════════════════════════════════════════════════════
# UYGULAMA BAŞLANGICI
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🔍 Canlı Ağ Trafiği Dinleniyor... (Durdurmak için Ctrl+C)")
    print("=" * 75)
    try:
        # Hatalı L3socket parametresi kaldırıldı, standart sniff kullanımı:
        sniff(prn=paket_islem, store=False)
    except KeyboardInterrupt:
        print("\n🛑 Canlı dinleme durduruldu.")
    except PermissionError:
        print("\n❌ HATA: Terminali YÖNETİCİ (Admin) olarak çalıştırmalısınız!")