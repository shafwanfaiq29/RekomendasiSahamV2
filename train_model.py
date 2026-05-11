"""
train_model.py — Script pelatihan model Random Forest untuk GoldStock Insight

Jalankan SEKALI sebelum web app digunakan:
    python train_model.py

Script ini akan:
1. Mengambil data historis harga saham ANTM.JK, MDKA.JK, BRMS.JK dari yfinance
2. Mengambil data emas GC=F dari yfinance
3. Melakukan scraping berita historis & menghitung sentimen
4. Menggabungkan dengan data fundamental dari CSV
5. Melatih model Random Forest
6. Menyimpan model ke models/rf_model.pkl
"""

import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import feedparser
import urllib.parse
import yfinance as yf

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─── Tambahkan parent folder ke path agar bisa import utils ───────────────────
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.sentiment import analyze_sentiment, _score_title
from utils.feature_engineering import calculate_technical_features

# ─── Konfigurasi ───────────────────────────────────────────────────────────────
TICKERS = {"ANTM": "ANTM.JK", "MDKA": "MDKA.JK", "BRMS": "BRMS.JK", "PSAB": "PSAB.JK", "ACES": "ACES.JK"}
GOLD_TICKER = "GC=F"
PERIOD = "2y"   # Ambil 2 tahun data historis untuk training

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
FUNDAMENTAL_FILE = os.path.join(DATA_DIR, "fundamental_clean.csv")
MODEL_FILE = os.path.join(MODEL_DIR, "rf_model.pkl")

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "MA7", "MA30", "Volatility",
    "Gold_Close", "Gold_Return",
    "Sentiment_Score", "News_Count",
    "PBV_x_ROE", "Price_to_Equity_Discount", "Relative_PE_ratio", 
    "EPS_Growth", "Debt_to_Total_Assets_Ratio", "Liquidity_Differential", 
    "CCE", "Operating_Efficiency", "Dividend_Payout", 
    "Yearly_Price_Change", "Composite_Rank", "Net_Debt_to_Equity"
]
TARGET_COL = "Target_Return"

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
KEYWORDS_MAP = {
    "ANTM": ["ANTM saham", "Antam emas"],
    "MDKA": ["MDKA saham", "Merdeka Copper Gold saham"],
    "BRMS": ["BRMS saham", "Bumi Resources Minerals saham"],
    "PSAB": ["PSAB saham", "J Resources Asia Pasifik saham"],
    "ACES": ["ACES saham", "Ace Hardware saham", "ACES retail"]
}


# ─── Fungsi Helper ─────────────────────────────────────────────────────────────

def fetch_stock_history(ticker_full: str, period: str = PERIOD) -> pd.DataFrame:
    print(f"  Mengambil data saham: {ticker_full} ...")
    try:
        raw = yf.download(ticker_full, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            print(f"  [WARN] Data kosong untuk {ticker_full}")
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df.reset_index(inplace=True)
        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df.dropna(subset=["Close"], inplace=True)
        return df
    except Exception as e:
        print(f"  [ERROR] {e}")
        return pd.DataFrame()


def fetch_gold_history(period: str = PERIOD) -> pd.DataFrame:
    print(f"  Mengambil data emas: {GOLD_TICKER} ...")
    try:
        raw = yf.download(GOLD_TICKER, period=period, auto_adjust=True, progress=False)
        if raw.empty:
            return pd.DataFrame()

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)

        df = raw[["Close"]].copy()
        df.index.name = "Date"
        df.reset_index(inplace=True)
        df.rename(columns={"Close": "Gold_Close"}, inplace=True)
        df["Gold_Close"] = pd.to_numeric(df["Gold_Close"], errors="coerce")
        df["Gold_Return"] = df["Gold_Close"].pct_change()
        df.dropna(subset=["Gold_Close"], inplace=True)
        return df
    except Exception as e:
        print(f"  [ERROR] mengambil emas: {e}")
        return pd.DataFrame()


def fetch_sentiment_for_ticker(ticker_short: str) -> float:
    """Scraping berita sekarang dan menghitung skor sentimen rata-rata."""
    keywords = KEYWORDS_MAP.get(ticker_short.upper(), [ticker_short + " saham"])
    all_titles = []

    for keyword in keywords:
        try:
            encoded = urllib.parse.quote(keyword)
            url = GOOGLE_NEWS_RSS.format(query=encoded)
            feed = feedparser.parse(url)
            for entry in feed.entries[:15]:
                title = entry.get("title", "")
                if title:
                    all_titles.append(title)
            time.sleep(0.5)
        except Exception as e:
            print(f"  [WARN] Gagal ambil berita '{keyword}': {e}")

    if not all_titles:
        return 0.0

    scores = [_score_title(t) for t in all_titles]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


# ─── Main Training Pipeline ────────────────────────────────────────────────────

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load fundamental
    print("\n[1] Memuat data fundamental...")
    if not os.path.exists(FUNDAMENTAL_FILE):
        print(f"  [ERROR] File tidak ditemukan: {FUNDAMENTAL_FILE}")
        sys.exit(1)

    fundamental_df = pd.read_csv(FUNDAMENTAL_FILE)
    fundamental_df["Ticker"] = fundamental_df["Ticker"].str.upper().str.strip()
    print(f"  OK — {len(fundamental_df)} baris fundamental dimuat")

    # 2. Ambil data emas
    print("\n[2] Mengambil data emas global...")
    gold_df = fetch_gold_history()
    if gold_df.empty:
        print("  [WARN] Data emas tidak tersedia, akan menggunakan 0")
        gold_df = pd.DataFrame(columns=["Date", "Gold_Close", "Gold_Return"])

    # 3. Ambil sentimen per ticker (scraping real)
    print("\n[3] Mengambil sentimen berita (real scraping)...")
    sentiment_map = {}
    news_count_map = {}
    for ticker_short in TICKERS:
        print(f"  -> {ticker_short}...")
        score = fetch_sentiment_for_ticker(ticker_short)
        sentiment_map[ticker_short] = score
        print(f"     Sentiment_Score = {score:.4f}")
        # Kita tidak tahu jumlah historis per hari, pakai nilai rata-rata
        news_count_map[ticker_short] = 10  # asumsi 10 berita/hari

    # 4. Proses tiap saham
    print("\n[4] Membangun dataset training...")
    all_frames = []

    for ticker_short, ticker_full in TICKERS.items():
        print(f"\n  -- {ticker_short} -------------------------")

        # Ambil data saham
        stock_df = fetch_stock_history(ticker_full)
        if stock_df.empty:
            print(f"  [SKIP] Tidak ada data untuk {ticker_short}")
            continue

        # Hitung fitur teknikal
        df = calculate_technical_features(stock_df)

        # Merge emas
        if not gold_df.empty:
            df["Date"] = pd.to_datetime(df["Date"]).dt.date
            gold_copy = gold_df.copy()
            gold_copy["Date"] = pd.to_datetime(gold_copy["Date"]).dt.date
            df = pd.merge(df, gold_copy[["Date", "Gold_Close", "Gold_Return"]],
                         on="Date", how="left")
            df["Gold_Close"] = df["Gold_Close"].ffill().fillna(0)
            df["Gold_Return"] = df["Gold_Return"].ffill().fillna(0)
        else:
            df["Gold_Close"] = 0.0
            df["Gold_Return"] = 0.0

        # Tambahkan fitur sentimen (gunakan skor saat ini untuk seluruh baris historis)
        # Ini pendekatan yang pragmatis untuk data historis
        df["Sentiment_Score"] = sentiment_map.get(ticker_short, 0.0)
        df["News_Count"] = news_count_map.get(ticker_short, 10)

        # Tambahkan fitur fundamental (statis per ticker)
        fund_row = fundamental_df[fundamental_df["Ticker"] == ticker_short]
        if fund_row.empty:
            print(f"  [WARN] Fundamental tidak ditemukan untuk {ticker_short}, skip")
            continue

        fund_dict = fund_row.iloc[0].to_dict()
        for col in ["PBV_x_ROE", "Price_to_Equity_Discount", "Relative_PE_ratio", "EPS_Growth", "Debt_to_Total_Assets_Ratio", "Liquidity_Differential", "CCE", "Operating_Efficiency", "Dividend_Payout", "Yearly_Price_Change", "Composite_Rank", "Net_Debt_to_Equity"]:
            df[col] = pd.to_numeric(fund_dict.get(col, 0), errors="coerce")

        # Target: return hari berikutnya (future return)
        df[TARGET_COL] = df["Return"].shift(-1)

        # Drop baris yang tidak lengkap
        df.dropna(subset=FEATURE_COLS + [TARGET_COL], inplace=True)

        print(f"  {len(df)} baris siap training")
        all_frames.append(df)

    if not all_frames:
        print("\n[ERROR] Tidak ada data training yang berhasil dibuat!")
        sys.exit(1)

    full_df = pd.concat(all_frames, ignore_index=True)
    print(f"\n  Total dataset: {len(full_df)} baris dari {len(all_frames)} ticker")

    # 5. Split dan training
    print("\n[5] Melatih model Random Forest...")
    X = full_df[FEATURE_COLS]
    y = full_df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 6. Evaluasi
    print("\n[6] Evaluasi model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:.6f} ({mae*100:.4f}%)")
    print(f"  RMSE : {rmse:.6f} ({rmse*100:.4f}%)")
    print(f"  R2   : {r2:.4f}")

    # 7. Simpan model
    print(f"\n[7] Menyimpan model ke: {MODEL_FILE}")
    joblib.dump(model, MODEL_FILE)
    print("  [OK] Model berhasil disimpan!")

    print("\n" + "="*50)
    print("Training selesai! Sekarang jalankan: streamlit run app.py")
    print("="*50)


if __name__ == "__main__":
    main()
