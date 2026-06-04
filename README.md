# GoldStock Insight 🟡 Premium V2

**Sistem Rekomendasi Saham Sektor Emas Berbasis Hybrid AI (GRU + XGBoost), Sentimen Berita, dan Fundamental**

GoldStock Insight V2 adalah web app *decision support system* premium untuk saham sektor emas di Indonesia (ANTM, MDKA, BRMS, PSAB). Aplikasi ini membantu investor mengambil keputusan investasi yang lebih objektif, terukur, dan berbasis data menggunakan pendekatan tiga pilar:
- **Pilar 1: Machine Learning (GRU + XGBoost)** untuk memprediksi return saham berdasarkan harga historis dan emas global.
- **Pilar 2: Sentimen Berita (NLP)** untuk menangkap sentimen pasar terbaru via Google News RSS.
- **Pilar 3: Analisis Fundamental** menggunakan *Piotroski F-Score* dan valuasi (Graham Price, PBV x ROE).

Seluruh metrik dari 3 pilar kemudian digabungkan melalui **Fuzzy Logic (Mamdani)** untuk menghasilkan rekomendasi akhir yang akurat dan mudah dipahami.

## Fitur Utama

- **Premium Hybrid Dashboard**: Visualisasi gaya terminal Bloomberg/premium interaktif untuk analisis saham yang komprehensif.
- **AI Engine (GRU + XGBoost Stacking)**: Model time-series canggih yang mempertimbangkan volatilitas harga saham dan pergerakan harga emas dunia.
- **Real-time News Sentiment**: Scraping sentimen market & saham dari Google News.
- **Risk & Hype Detector**: Mendeteksi jika saham sedang *overhyped* (FOMO) atau memiliki risiko volatilitas tinggi.
- **Investment Simulator**: Mengestimasi nilai masa depan berdasarkan rekomendasi AI dan input modal Anda.
- **Explainable AI**: Sistem menjelaskan mengapa sebuah rekomendasi diberikan dengan bahasa yang mudah dipahami pemula.

## Persyaratan Sistem

- Python 3.9+
- TensorFlow >= 2.12.0
- XGBoost >= 1.7.0
- Streamlit >= 1.28.0
- Koneksi internet (untuk scraping data real-time via yfinance dan Google News RSS)

## Cara Instalasi dan Penggunaan

1. **Clone repositori ini atau ekstrak folder proyek:**
   ```bash
   cd "coba bangun"
   ```

2. **Buat dan aktifkan virtual environment (opsional namun direkomendasikan):**
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Siapkan Data & Model Machine Learning:**
   Pastikan folder `models/` berisi file model GRU (`.h5`), XGBoost (`.json`), dan Scaler (`.pkl`) untuk setiap saham (ANTM, BRMS, MDKA, PSAB).
   Pastikan folder `data/` berisi data sentimen dan evaluasi fundamental (`fundamental_clean.csv`).

5. **Jalankan Aplikasi Streamlit:**
   ```bash
   streamlit run app.py
   ```

6. Buka browser dan akses URL lokal yang tertera (biasanya `http://localhost:8501`).

## Struktur Proyek

```
coba bangun/
│
├── app.py                  # File utama aplikasi Streamlit (Premium V2)
├── requirements.txt        # Daftar dependensi library Python
├── BACKEND_FUNCTION_AUDIT.md # Dokumentasi audit fungsi redesign
├── README.md               # File dokumentasi ini
│
├── data/                   # Data CSV Fundamental & Sentimen
├── models/                 # Model GRU (.h5), XGB (.json), dan Scaler (.pkl)
└── utils/                  # Modul-modul fungsi spesifik (opsional/development)
```

## Disclaimer

Aplikasi ini dibangun untuk keperluan edukasi dan *Capstone Project*. Semua rekomendasi, prediksi, simulasi return, dan skor yang dihasilkan **bukanlah nasihat keuangan (financial advice)**. Segala keputusan investasi sepenuhnya merupakan tanggung jawab pengguna.
