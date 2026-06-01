# Catatan Penjelasan Kode — GoldStock Insight

Dokumen ini menjelaskan **seluruh kode proyek** secara detail dan mudah dipahami. Cocok untuk sidang capstone, onboarding tim, atau revisi sebelum presentasi.

> **Cara pakai:** Baca bagian [Ringkasan](#1-ringkasan-sistem) dulu, lalu [Alur Data](#2-alur-data-dari-klik-tombol-sampai-rekomendasi), baru detail per file.

---

## Daftar Isi

1. [Ringkasan Sistem](#1-ringkasan-sistem)
2. [Alur Data (dari klik tombol sampai rekomendasi)](#2-alur-data-dari-klik-tombol-sampai-rekomendasi)
3. [Struktur Folder](#3-struktur-folder)
4. [File Utama](#4-file-utama)
   - [app.py](#41-apppy--aplikasi-web-streamlit)
   - [train_model.py](#42-train_modelpy--pelatihan-model)
5. [Folder utils/](#5-folder-utils)
6. [Folder data/](#6-folder-data)
7. [Folder models/](#7-folder-models)
8. [Tiga Pilar Analisis](#8-tiga-pilar-analisis)
9. [Glosarium Istilah](#9-glosarium-istilah)
10. [Tips Debug & Pengembangan](#10-tips-debug--pengembangan)

---

## 1. Ringkasan Sistem

**GoldStock Insight** adalah aplikasi web (Streamlit) yang membantu investor menganalisis saham sektor emas Indonesia:

| Saham | Kode Bursa | Perusahaan |
|-------|------------|------------|
| ANTM | ANTM.JK | Aneka Tambang |
| MDKA | MDKA.JK | Merdeka Copper Gold |
| BRMS | BRMS.JK | Bumi Resources Minerals |
| PSAB | PSAB.JK | J Resources Asia Pasifik |

Sistem menggabungkan **3 sumber keputusan**:

| Pilar | Sumber | Output utama |
|-------|--------|----------------|
| **1. Teknikal & Prediksi** | yfinance + model ML (XGBoost/GRU/Meta atau fallback) | `predicted_return` |
| **2. Sentimen** | Google News RSS + IndoBERT (CSV) atau rule-based | `sentiment_score` |
| **3. Fundamental** | CSV Piotroski/Graham + `fundamental_clean.csv` | `Composite_Rank`, harga wajar |

Rekomendasi akhir memakai **logika fuzzy (Mamdani)** yang menggabungkan ketiga pilar menjadi skor 0–100, lalu label: *Jangka Panjang*, *Jangka Pendek*, atau *Overhyped / Hindari*.

---

## 2. Alur Data (dari klik tombol sampai rekomendasi)

```
┌─────────────────────────────────────────────────────────────────┐
│  USER: Pilih saham + tujuan investasi → klik "Analisis Saham"   │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. load_fundamental_data()  → baca data/fundamental_clean.csv   │
│  2. fetch_stock_data()       → harga saham dari yfinance        │
│  3. fetch_gold_data()        → harga emas GC=F dari yfinance    │
│  4. fetch_news()             → berita Google News RSS           │
│  5. apply_sentiment()        → skor sentimen dari judul berita  │
│     (+ injeksi CSV IndoBERT jika ada di KAGGLE_PILAR_DATA)      │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  prepare_latest_row() → gabung harga, emas, sentimen, fundamental│
│  predict_return()     → model stacking atau rule-based fallback │
│  generate_recommendation() → fuzzy Mamdani → label + skor       │
│  calculate_risk_level() / detect_overhyped_status() → tambahan UI │
└───────────────────────────────┬─────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│  Tampilan: kartu rekomendasi, grafik, tab berita, fundamental,  │
│  risk & hype, compare, market news, watchlist                   │
└─────────────────────────────────────────────────────────────────┘
```

**Tanpa klik analisis** (halaman awal): sistem otomatis memanggil `generate_watchlist()` → menampilkan *Top Pick* dan tabel perbandingan semua saham.

---

## 3. Struktur Folder

```
coba bangun/
├── app.py                      ← Aplikasi utama (UI + sebagian besar logika)
├── train_model.py              ← Script latih Random Forest (sekali jalan)
├── PENJELASAN_KODE.md          ← File ini
├── requirements.txt            ← Dependensi Python
├── README.md, PRD.md, Claude.md
│
├── data/
│   ├── fundamental_clean.csv           ← Rasio fundamental untuk ML & UI
│   ├── fundamental_evaluasi_final.csv  ← Graham + Piotroski (hasil Kaggle)
│   └── sentimen_{TICKER}.csv           ← Skor IndoBERT per emiten
│
├── models/
│   ├── rf_model.pkl                    ← Random Forest (dari train_model.py)
│   ├── model_xgb_{TICKER}.JK.pkl       ← Base model XGBoost (opsional)
│   ├── model_gru_{TICKER}.JK.pkl       ← Base model GRU (opsional)
│   └── meta_model_{TICKER}.JK.pkl      ← Meta learner stacking (opsional)
│
└── utils/                      ← Modul terpisah (bisa dipanggil dari script lain)
    ├── fetch_stock.py
    ├── fetch_news.py
    ├── sentiment.py
    ├── feature_engineering.py
    ├── recommendation.py
    ├── data_engine.py
    ├── fuzzy_engine.py
    └── decision_support.py
```

**Catatan penting:** Banyak fungsi di `utils/` **juga ditulis ulang di `app.py`** agar UI bisa cache dengan `@st.cache_data` dan tidak perlu import berulang. Untuk analisis harian, **`app.py` adalah sumber kebenaran utama** saat aplikasi dijalankan.

---

## 4. File Utama

### 4.1 `app.py` — Aplikasi Web Streamlit

File terbesar (~2200 baris). Berisi: tampilan UI, pengambilan data, prediksi, rekomendasi, dan fitur tambahan (watchlist, market news, compare).

#### Bagian A — Konfigurasi halaman & CSS (baris ~17–423)

| Kode | Fungsi |
|------|--------|
| `st.set_page_config(...)` | Judul tab browser, layout lebar, sidebar disembunyikan |
| `st.markdown("<style>...</style>")` | Tema gelap + aksen emas: hero, kartu metrik, badge rekomendasi, responsif mobile |

#### Bagian B — Konstanta global (baris ~430–488)

| Variabel | Penjelasan |
|----------|------------|
| `TICKER_MAP` | Map kode pendek → kode Yahoo (`ANTM` → `ANTM.JK`) |
| `COMPANY_NAMES` | Nama lengkap perusahaan untuk teks penjelasan |
| `NEWS_KEYWORDS` | Daftar kata kunci pencarian berita per saham |
| `MODEL_FEATURES` | 24 kolom input yang **harus sama urutannya** saat training & prediksi |
| `MARKET_NEWS_CATEGORIES` | Kategori berita pasar (Emas, IHSG, Ekonomi, dll.) |
| `MARKET_POSITIVE_WORDS` / `MARKET_NEGATIVE_WORDS` | Kamus sentimen khusus berita makro |
| `STOPWORDS_ID` | Kata diabaikan saat hitung *trending topics* |

#### Bagian C — Helper tampilan (baris ~495–535)

| Fungsi | Fungsi |
|--------|--------|
| `format_percent(value)` | Ubah desimal → string persen, mis. `0.05` → `5.00%` |
| `format_number(value)` | Format angka besar (T = triliun, B = miliar, M = juta) |
| `metric_card(label, value, small_text)` | Render kartu metrik HTML custom |
| `get_badge_class(recommendation)` | Warna badge: hijau/biru/oranye/merah sesuai rekomendasi |

#### Bagian D — Pengambilan data (baris ~542–661)

Semua memakai **`@st.cache_data`** agar tidak fetch ulang setiap kali Streamlit rerun.

| Fungsi | Input | Output | Penjelasan singkat |
|--------|-------|--------|-------------------|
| `load_fundamental_data()` | — | DataFrame | Baca `fundamental_clean.csv`; jika tidak ada, pakai data fallback hardcoded |
| `fetch_stock_data(ticker_code, period)` | `ANTM.JK`, `2y` | DataFrame OHLCV + MA7, MA30, Return, Volatility | Download dari yfinance; flatten MultiIndex kolom |
| `fetch_gold_data(period)` | `2y` | Date, Gold_Close, Gold_Return | Emas kontrak `GC=F` |
| `fetch_news(ticker)` | `ANTM` | Date, Title, Source, Link | Loop keyword → RSS Google News → deduplikasi judul |

**Rumus teknikal di `fetch_stock_data`:**

- `Return` = perubahan harga harian: `(Close_hari_ini - Close_kemarin) / Close_kemarin`
- `MA7` / `MA30` = rata-rata harga 7 / 30 hari (indikator tren)
- `Volatility` = standar deviasi Return 7 hari (ukuran risiko harga)

#### Bagian E — Sentimen berita saham (baris ~668–722)

| Fungsi | Penjelasan |
|--------|------------|
| `get_sentiment_score(text)` | Hitung kata positif/negatif di judul → skor -1 s/d 1 |
| `apply_sentiment(news_df)` | Terapkan ke semua baris berita → rata-rata skor + label Positive/Neutral/Negative |

**Logika label:** rata-rata > 0.1 → Positive; < -0.1 → Negative; selain itu Neutral.

#### Bagian F — Berita pasar & tren (baris ~729–872)

| Fungsi | Penjelasan |
|--------|------------|
| `fetch_market_news(category)` | Ambil berita makro sesuai kategori (Emas, IHSG, dll.) |
| `apply_market_sentiment(news_df)` | Sentimen khusus pasar (kamus `MARKET_*_WORDS`) |
| `get_trending_topics(news_df)` | Kata paling sering muncul di judul (minus stopwords) |
| `create_market_sentiment_chart` | Pie chart Plotly distribusi sentimen |
| `create_category_bar_chart` | Bar chart jumlah berita per kategori |
| `generate_market_insight(...)` | Paragraf insight otomatis berbahasa Indonesia |

#### Bagian G — Data Kaggle & Fuzzy (baris ~884–971)

| Fungsi | Penjelasan |
|--------|------------|
| `ambil_data_asli_kaggle()` | Baca CSV fundamental + sentimen IndoBERT per ticker; hasil disimpan di `KAGGLE_PILAR_DATA` |
| `eksekusi_fuzzy_mamdani(pred_return, sentimen, fundamental)` | Mesin fuzzy: 3 input → skor rekomendasi 0–100 |

**Aturan fuzzy (ringkas):**

- Fundamental *sakit* ATAU sentimen *negatif* → cenderung **hindari**
- Fundamental *sehat* + return *bullish* + sentimen *positif* → **jangka panjang**
- Fundamental *biasa* + sentimen *positif* → **jangka pendek**
- Return *bearish* + fundamental *sakit* → **hindari**
- Return *bullish* + fundamental *sehat* → **jangka panjang**

Input di-clip ke rentang aman sebelum `simulasi.compute()`.

#### Bagian H — Prediksi & rekomendasi (baris ~983–1215)

| Fungsi | Penjelasan |
|--------|------------|
| `load_model_dinamis(ticker)` | Load 3 file: XGB + GRU + meta model per ticker (jika ada di folder `models/`) |
| `predict_return(ticker, latest_row)` | Prediksi stacking: XGB & GRU → meta model; gagal → `fallback_predict_return` |
| `fallback_predict_return(row)` | **Tanpa ML:** gabung sinyal MA, sentimen, fundamental, emas, penalty volatilitas |
| `prepare_latest_row(...)` | Merge saham + emas + sentimen + kolom fundamental → satu baris terakhir |
| `generate_recommendation(...)` | Panggil fuzzy; skor ≥65 → Jangka Panjang; ≥40 → Jangka Pendek; else Overhyped/Hindari |
| `generate_explanation(...)` | Teks narasi rekomendasi untuk tab "Alasan Rekomendasi" |
| `calculate_risk_level(...)` | Skor risiko 0–100 → Low / Medium / High Risk |
| `detect_overhyped_status(...)` | Deteksi FOMO: sentimen tinggi + banyak berita + return tinggi tapi fundamental lemah |

**Mapping skor fuzzy → label:**

```
skor_fuzzy >= 65  → "Jangka Panjang"
skor_fuzzy >= 40  → "Jangka Pendek"
skor_fuzzy < 40   → "Overhyped / Hindari"
```

#### Bagian I — Watchlist & perbandingan (baris ~1222–1336)

| Fungsi | Penjelasan |
|--------|------------|
| `generate_watchlist()` | Loop semua ticker: fetch data → prediksi → rekomendasi → satu baris per saham |
| `get_top_recommendation(watchlist_df)` | Filter: return > 0, bukan High Risk, bukan Overhyped → skor tertinggi |
| `compare_stocks(t1, t2, goal, df)` | Bandingkan dua saham; bobot berbeda untuk Jangka Panjang vs Pendek |

#### Bagian J — Komponen UI (baris ~1344–1562)

| Fungsi | Penjelasan |
|--------|------------|
| `render_market_news()` | Section berita pasar lengkap dengan chart & daftar berita |
| `render_watchlist(df)` | Tabel ringkasan semua saham |
| `create_stock_chart(df, ticker)` | Line chart Close + MA7 + MA30 (Plotly) |
| `create_sentiment_chart(news_df)` | Donut chart sentimen berita emiten |

#### Bagian K — Alur utama UI (baris ~1569–akhir)

1. **Hero** — judul & deskripsi produk  
2. **Input** — selectbox saham & tujuan investasi + tombol analisis  
3. **Jika tombol TIDAK diklik:** Top Pick + Watchlist + Market News + "Cara Kerja" → `st.stop()`  
4. **Jika tombol diklik:** spinner → pipeline analisis → kartu rekomendasi → 8 tab:
   - Harga Saham | Sentimen | Fundamental | Alasan | Risk & Hype | Compare | Market News | Watchlist  
5. **Disclaimer** — bukan ajakan beli/jual saham  

---

### 4.2 `train_model.py` — Pelatihan Model

Script **dijalankan sekali** (atau saat perlu retrain), bukan saat user buka web.

**Alur `main()`:**

| Langkah | Apa yang dilakukan |
|---------|-------------------|
| 1 | Load `fundamental_clean.csv` |
| 2 | Download historis emas `GC=F` 2 tahun |
| 3 | Scrape sentimen berita **saat ini** per ticker (bukan historis per hari) |
| 4 | Per ticker: download saham → `calculate_technical_features` → merge emas → isi sentimen & fundamental → target = return **hari berikutnya** (`shift(-1)`) |
| 5 | Gabung semua ticker → `train_test_split` 80/20 |
| 6 | Latih `RandomForestRegressor` (200 pohon, max_depth 10) |
| 7 | Cetak MAE, RMSE, R² |
| 8 | Simpan `models/rf_model.pkl` |

**Fungsi pendukung:**

| Fungsi | Penjelasan |
|--------|------------|
| `fetch_stock_history(ticker_full)` | Ambil OHLCV historis untuk training |
| `fetch_gold_history()` | Ambil emas historis |
| `fetch_sentiment_for_ticker(ticker_short)` | RSS + rata-rata `_score_title` dari utils |

**Keterbatasan desain:** Sentimen yang sama diisi ke **semua baris historis** per ticker (pragmatis karena tidak ada arsip berita harian). Untuk sidang, sebutkan sebagai asumsi/limitasi.

---

## 5. Folder `utils/`

Modul-modul ini dipakai terutama oleh `train_model.py` dan bisa dipakai ulang jika `app.py` direfaktor.

### 5.1 `fetch_stock.py`

| Fungsi | Parameter | Return |
|--------|-----------|--------|
| `fetch_stock_data(ticker_short, period)` | `ANTM`, `6mo` | DataFrame OHLCV + kolom Ticker |
| `fetch_gold_price(period)` | `6mo` | DataFrame Gold_Close, Gold_Return |

- Error handling: `try/except` → DataFrame kosong + print log  
- Menangani **MultiIndex** kolom yfinance  

### 5.2 `fetch_news.py`

| Fungsi | Penjelasan |
|--------|------------|
| `fetch_news(ticker_short, max_per_keyword)` | Loop `KEYWORDS_MAP` → parse RSS → deduplikasi judul → sort by date |

- Delay `time.sleep(0.5)` antar keyword untuk hindari rate limit Google  

### 5.3 `sentiment.py`

| Fungsi | Penjelasan |
|--------|------------|
| `_score_title(title)` | Skor satu judul; kata berat (`melesat`, `bangkrut`) bobot 1.5 |
| `analyze_sentiment(news_df)` | Rata-rata skor + label Positif/Netral/Negatif (threshold ±0.1) |

Kamus lebih lengkap daripada versi di `app.py` (istilah keuangan & emas).

### 5.4 `feature_engineering.py`

| Fungsi | Penjelasan |
|--------|------------|
| `calculate_technical_features(stock_df)` | Return, MA7, MA30, Volatility |
| `merge_gold_features(stock_df, gold_df)` | Left join tanggal; forward-fill jika emas libur |
| `build_feature_row(stock_df, gold_df, sentiment, fundamental)` | Satu baris terakhir siap `model.predict()` |

### 5.5 `recommendation.py`

Logika rekomendasi **berbasis aturan PRD** (bukan fuzzy):

| Kondisi | Rekomendasi |
|---------|-------------|
| Predicted return < 3% | Tidak Disarankan |
| 3% ≤ return ≤ 7% | Jangka Pendek |
| return > 7% & fundamental ≥ 0.70 | Jangka Panjang |
| return > 7% & fundamental < 0.70 | Overhyped / Hindari |
| Sentimen < -0.3 | Turunkan 1 level |

Juga berisi `calculate_risk_level` dan `detect_overhyped_status` (mirip `app.py`).

> **Saat runtime Streamlit:** `app.py` memakai **fuzzy**, bukan file ini — kecuali Anda mengubah import.

### 5.6 `data_engine.py`

| Item | Penjelasan |
|------|------------|
| `MASTER_STOCK_DATA` | Dictionary statis hasil 3 notebook Kaggle: pred_return, sentiment, Graham, Piotroski, harga pasar, D/E, volatilitas |
| `dapatkan_data_emiten(ticker)` | Ambil entry by ticker (`ANTM` dari `ANTM.JK`) |

Digunakan oleh `decision_support.py` sebagai sumber data offline.

### 5.7 `fuzzy_engine.py`

| Fungsi | Penjelasan |
|--------|------------|
| `hitung_rekomendasi_fuzzy(...)` | Versi modular dari `eksekusi_fuzzy_mamdani` di app.py — logika identik |

Library: `scikit-fuzzy` (`skfuzzy`).

### 5.8 `decision_support.py`

| Fungsi | Penjelasan |
|--------|------------|
| `deteksi_overhyped(data_emiten)` | Jika harga pasar / Graham > 1.3 dan sentimen positif → FOMO bubble |
| `hitung_risk_level(data_emiten)` | HIGH jika D/E > 1 atau volatilitas > 25% |
| `bangun_dashboard_metrics()` | Watchlist dari `MASTER_STOCK_DATA` + fuzzy |

**Alternatif pipeline** jika tidak pakai fetch real-time — cocok untuk demo offline.

### 5.9 `__init__.py`

Hanya komentar paket; tidak meng-export simbol khusus.

---

## 6. Folder `data/`

| File | Isi | Dipakai oleh |
|------|-----|--------------|
| `fundamental_clean.csv` | Rasio valuasi & kesehatan keuangan per ticker (PBV×ROE, PE discount, Composite_Rank, dll.) | `load_fundamental_data`, training, prediksi |
| `fundamental_evaluasi_final.csv` | Harga wajar Graham, Piotroski F-Score, skor fuzzy Piotroski | `ambil_data_asli_kaggle()` |
| `sentimen_ANTM.csv`, `sentimen_MDKA.csv`, … | Skor IndoBERT (`Skor_Sentimen_Final`) | Override sentimen di `KAGGLE_PILAR_DATA` |

**Catatan:** Nama file sentimen di `app.py` mencari `sentimen_{TICKER}.JK.csv`, sedangkan di repo ada `sentimen_ANTM.csv` (tanpa `.JK`). Pastikan nama file konsisten agar injeksi IndoBERT aktif.

---

## 7. Folder `models/`

| Pola file | Peran |
|-----------|--------|
| `rf_model.pkl` | Random Forest regressor (output `train_model.py`) |
| `model_xgb_{TICKER}.JK.pkl` | Model base XGBoost |
| `model_gru_{TICKER}.JK.pkl` | Model base GRU (bisa butuh format tensor khusus) |
| `meta_model_{TICKER}.JK.pkl` | Meta-learner: input = [pred_xgb, pred_gru] → final return |
| `xgb_*.json`, `gru_*.h5`, `scaler_*.pkl` | Artefak training Kaggle (belum semua di-wire ke app) |

**Stacking di `predict_return`:**

```
Fitur (24 kolom) ──► XGB ──┐
                           ├──► Meta Model ──► predicted_return
Fitur (24 kolom) ──► GRU ──┘
```

Jika salah satu file hilang → fallback rule-based di `fallback_predict_return`.

---

## 8. Tiga Pilar Analisis

```
        ┌──────────────┐
        │   PILAR 1    │  Prediksi return (ML / rule-based)
        │   Teknikal   │  MA7, MA30, volatilitas, emas GC=F
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │   PILAR 2    │  Sentimen berita (-1 … +1)
        │   Sentimen   │  RSS real-time atau CSV IndoBERT
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │   PILAR 3    │  Fundamental / Piotroski fuzzy
        │  Fundamental │  CSV + Graham undervalued check
        └──────┬───────┘
               │
        ┌──────▼───────┐
        │ FUZZY ENGINE │  Skor 0–100 → Rekomendasi akhir
        └──────────────┘
```

---

## 9. Glosarium Istilah

| Istilah | Arti dalam proyek ini |
|---------|------------------------|
| **Return** | Persentase perubahan harga dari hari ke hari |
| **MA7 / MA30** | Moving Average — rata-rata harga 7/30 hari |
| **Volatility** | Seberapa liar pergerakan harga (std return) |
| **GC=F** | Ticker Yahoo untuk emas berjangka COMEX |
| **Composite_Rank** | Skor gabungan rasio fundamental (semakin tinggi semakin baik) |
| **Piotroski F-Score** | Skor 0–9 kesehatan keuangan perusahaan |
| **Graham Price** | Estimasi harga wajar menurut Benjamin Graham |
| **Stacking** | Dua model memprediksi, model ketiga menggabungkan prediksi |
| **Fuzzy Mamdani** | Sistem keputusan berbasis aturan kabur (bukan hitam-putih) |
| **Overhyped** | Harga/sentimen terlalu optimis tanpa fundamental mendukung |
| **FOMO** | Fear Of Missing Out — beli karena hype, bukan analisis |
| **yfinance** | Library Python untuk download data Yahoo Finance |
| **RSS** | Feed berita XML dari Google News |

---

## 10. Tips Debug & Pengembangan

### Menjalankan aplikasi

```bash
venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

### Melatih ulang Random Forest

```bash
python train_model.py
```

### Checklist jika analisis gagal

1. **Internet** — yfinance & Google News butuh koneksi  
2. **File fundamental** — pastikan `data/fundamental_clean.csv` ada  
3. **Urutan kolom model** — `MODEL_FEATURES` harus sama dengan training  
4. **MultiIndex yfinance** — sudah di-handle dengan `droplevel` / `get_level_values`  
5. **Model per ticker** — cek ada `model_xgb_ANTM.JK.pkl` dll. jika mau stacking  

### Perbedaan dua jalur rekomendasi

| Aspek | `app.py` (aktif) | `utils/recommendation.py` |
|-------|------------------|---------------------------|
| Metode | Fuzzy Mamdani | Aturan threshold PRD |
| Input utama | KAGGLE_PILAR_DATA + prediksi live | Return % + fundamental score 0–1 |
| Label | Jangka Panjang / Pendek / Overhyped | + "Tidak Disarankan" |

### Ide perbaikan (opsional)

- Satukan logika duplikat `app.py` ↔ `utils/`  
- Perbaiki nama file CSV sentimen agar match `ambil_data_asli_kaggle`  
- Di `generate_watchlist`, pastikan variabel ticker konsisten (hindari referensi `selected_ticker` di luar scope)  
- Tambahkan logging instead of `print` di production  

---

## Lampiran — Daftar 24 Fitur Model (`MODEL_FEATURES`)

| No | Kolom | Asal |
|----|-------|------|
| 1–5 | Open, High, Low, Close, Volume | yfinance saham |
| 6–9 | Return, MA7, MA30, Volatility | Dihitung di fetch / feature engineering |
| 10–11 | Gold_Close, Gold_Return | yfinance GC=F |
| 12–13 | Sentiment_Score, News_Count | Berita / CSV IndoBERT |
| 14–24 | PBV_x_ROE, Price_to_Equity_Discount, … Net_Debt_to_Equity | fundamental_clean.csv |

**Target training (`train_model.py`):** `Target_Return` = return hari **berikutnya** (shift -1 pada kolom Return).

---

*Dokumen ini dibuat untuk proyek capstone GoldStock Insight. Perbarui jika struktur kode berubah.*
