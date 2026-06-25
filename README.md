# 💧 Analisis Debit Andalan — SNI 6738:2015

Aplikasi web untuk analisis **debit andalan** berbasis metode **F.J. Mock** (neraca air bulanan) sesuai **SNI 6738:2015**.

Dibangun dengan **Python + Streamlit** — bisa dijalankan lokal maupun di-deploy gratis ke [Streamlit Cloud](https://streamlit.io/cloud).

---

## ✨ Fitur

| Fitur | Keterangan |
|---|---|
| 📍 Multi-stasiun | Input beberapa stasiun hujan |
| ⚖️ Bobot Thiessen | Dihitung otomatis via IDW dari koordinat |
| 🌿 ETP Penman-Monteith | FAO-56, input iklim bulanan |
| 🌊 Model F.J. Mock | Simulasi neraca air 12 bulan (hingga kondisi tunak) |
| 📉 FDC | Kurva Durasi Aliran interaktif |
| 📊 Q80 / Q90 / Q95 | Debit andalan sesuai SNI 6738:2015 |
| 🔄 Log Pearson III | Debit kala ulang T=2,5,10,25,50,100 tahun |
| 💾 Ekspor | Download hasil CSV & JSON |

---

## 🗂️ Struktur File

```
debit-andalan/
├── streamlit_app.py    ← Aplikasi utama Streamlit
├── hydrology.py        ← Engine perhitungan (Mock, ETP, FDC, LP3)
├── requirements.txt    ← Dependensi Python
└── README.md
```

---

## 🚀 Cara Menjalankan

### A. Lokal (di komputer sendiri)

**Prasyarat:** Python 3.10+ sudah terinstal.

```bash
# 1. Clone repo ini
git clone https://github.com/USERNAME/debit-andalan.git
cd debit-andalan

# 2. (Opsional) Buat virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

# 3. Install dependensi
pip install -r requirements.txt

# 4. Jalankan
streamlit run streamlit_app.py
```

Buka browser ke **http://localhost:8501**

---

### B. Deploy ke Streamlit Cloud (gratis, online)

1. **Fork / upload** repo ini ke akun GitHub kamu
2. Buka [https://streamlit.io/cloud](https://streamlit.io/cloud) → **New app**
3. Pilih repo, branch `main`, file utama `streamlit_app.py`
4. Klik **Deploy** — selesai! Dapat URL publik otomatis

---

## 📥 Cara Pakai Aplikasi

1. **Muat Contoh Data** (opsional) untuk melihat hasil langsung
2. Tab **📍 Stasiun & Hujan**: tambah stasiun, isi koordinat & curah hujan bulanan
3. Tab **🌡️ Data Iklim**: isi suhu, kelembaban, radiasi, kecepatan angin per bulan
4. **Sidebar** kiri: atur parameter DAS (luas, koef. infiltrasi, kapasitas lengas tanah, dll.)
5. Tab **▶ Jalankan Analisis**: klik tombol untuk menghitung
6. Tab **📊 Hasil**: lihat grafik & tabel, download CSV/JSON

---

## 📐 Metode yang Digunakan

### Bobot Thiessen
Dihitung dengan metode IDW (Inverse Distance Weighting) berbasis jarak Haversine dari centroid DAS ke masing-masing stasiun.

### ETP Penman-Monteith (FAO-56)
ETP bulanan dihitung dari data iklim: suhu rata-rata (T), kelembaban relatif (RH), radiasi matahari (Rs), dan kecepatan angin (uz).

### Neraca Air F.J. Mock
Model simulasi bulanan yang menghitung:
- Evapotranspirasi aktual (AET)
- Surplus air
- Limpasan langsung (direct runoff)
- Infiltrasi & aliran dasar (baseflow)
- Tampungan air tanah (groundwater storage)

### Debit Andalan
Dari Kurva Durasi Aliran (FDC), diambil:
- **Q80** → untuk irigasi
- **Q90** → untuk PLTA
- **Q95** → untuk air baku

### Analisis Frekuensi
Distribusi **Log Pearson III** untuk menghitung debit kala ulang.

---

## 📋 Referensi

- SNI 6738:2015 — *Perhitungan Debit Andalan Sungai dengan Kurva Durasi Debit*
- Mock, F.J. (1973). *Land Capability Appraisal Indonesia*. FAO/UNDP
- Allen et al. (1998). *Crop Evapotranspiration*. FAO Irrigation Paper No. 56

---

## 📄 Lisensi

MIT License — bebas digunakan dan dimodifikasi.
