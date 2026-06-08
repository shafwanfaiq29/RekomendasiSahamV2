import os
import numpy as np
import pandas as pd
import yfinance as yf
import feedparser
from urllib.parse import quote
import time
from functools import lru_cache
from sklearn.preprocessing import MinMaxScaler
from config import NEWS_KEYWORDS, MARKET_NEWS_CATEGORIES

class DummyST:
    def cache_data(self, *args, **kwargs):
        return lru_cache(maxsize=32)

st = DummyST()

# ==============================================================================
# KONSTANTA
# ==============================================================================
_EXCEL_PATH = "data/Fundamental Saham Emas 2026-06-07.xlsx"
_TARGET_TICKERS = ["ANTM", "BRMS", "MDKA", "PSAB", "UNTR", "HRTA", "ARCI", "AMMN"]

# Kolom yang WAJIB ada di dict fundamental yang dihasilkan get_fundamental_row().
# Setiap kolom punya nilai default 0.0 sebagai fallback aman (tidak pernah ditampilkan
# ke user sebagai data valid — hanya agar app tidak crash).
_FUNDAMENTAL_DEFAULTS = {
    # === Dari sheet 'analysis' — fitur model ML ===
    "PBV_x_ROE":                  0.0,
    "Price_to_Equity_Discount":   0.0,
    "Relative_PE_ratio":          0.0,
    "EPS_Growth":                 0.0,
    "Debt_to_Total_Assets_Ratio": 0.0,
    "Liquidity_Differential":     0.0,
    "CCE":                        0.0,
    "Operating_Efficiency":       0.0,
    "Dividend_Payout":            0.0,
    "Yearly_Price_Change":        0.0,
    "Composite_Rank":             0.0,   # skala 0–1, untuk risk & hype
    "Net_Debt_to_Equity":         0.0,
    "Close_Price_At_Analysis":    0.0,
    # === Dari notebook (key-statistics → dihitung) ===
    "Harga_Wajar_Graham":         0.0,
    "Skor_Piotroski_Fuzzy":       0.0,   # skala -1/+1, untuk fuzzy Mamdani
    "Piotroski_F_Score":          0.0,
    "Current_EPS_TTM":            0.0,
    "Book_Value_Per_Share":       0.0,
}


# ==============================================================================
# LOAD & BUILD — single pipeline, dipanggil sekali saat startup
# ==============================================================================

def _load_from_excel() -> pd.DataFrame:
    """
    Gabungkan sheet 'analysis' + perhitungan notebook dari 'key-statistics'.

    Kolom output (semua 8 ticker, tanpa EMAS):
      Ticker
      --- dari sheet analysis ---
      PBV_x_ROE, Close_Price_At_Analysis, Price_to_Equity_Discount,
      Relative_PE_ratio, EPS_Growth, Debt_to_Total_Assets_Ratio,
      Liquidity_Differential, CCE, Operating_Efficiency, Dividend_Payout,
      Yearly_Price_Change, Composite_Rank, Net_Debt_to_Equity
      --- dari notebook pipeline (key-statistics) ---
      Current_EPS_TTM, Book_Value_Per_Share, Harga_Wajar_Graham,
      Piotroski_F_Score, Skor_Piotroski_Fuzzy
    """
    # ── Sheet analysis ──
    df_a = pd.read_excel(_EXCEL_PATH, sheet_name='analysis')
    df_a.columns = df_a.columns.str.strip()
    df_a = df_a[df_a['Ticker'].isin(_TARGET_TICKERS)].copy().reset_index(drop=True)
    df_a = df_a.rename(columns={
        'PBV x ROE':                    'PBV_x_ROE',
        'Close Price':                  'Close_Price_At_Analysis',
        'Price to Equity Discount (%)': 'Price_to_Equity_Discount',
        'Relative PE ratio (TTM)':      'Relative_PE_ratio',
        'EPS Growth':                   'EPS_Growth',
        'Debt to Total Assets Ratio':   'Debt_to_Total_Assets_Ratio',
        'Liquidity Differential':       'Liquidity_Differential',
        'CCE':                          'CCE',
        'Operating Efficiency':         'Operating_Efficiency',
        'Dividend Payout Efficiency':   'Dividend_Payout',
        'Yearly Price Change':          'Yearly_Price_Change',
        'Composite Rank':               'Composite_Rank',
        'Net Debt to Equity':           'Net_Debt_to_Equity',
    })

    # ── Sheet key-statistics + pipeline notebook ──
    df_ks = pd.read_excel(_EXCEL_PATH, sheet_name='key-statistics')
    df_ks.columns = df_ks.columns.str.strip()
    df_ks = df_ks[df_ks['Ticker'].isin(_TARGET_TICKERS)].copy().reset_index(drop=True)

    # Pipeline Cell [04]: Graham Number
    eps  = df_ks['Current EPS (TTM)'].fillna(0)
    bvps = df_ks['Current Book Value Per Share'].fillna(0)
    df_ks['Harga_Wajar_Graham'] = np.sqrt(np.clip(22.5 * eps * bvps, 0, None))

    # Pipeline Cell [06]: Piotroski Fuzzy — skala -1/+1
    if 'Piotroski F-Score' in df_ks.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        df_ks['Skor_Piotroski_Fuzzy'] = scaler.fit_transform(
            df_ks[['Piotroski F-Score']].fillna(0)
        )
    else:
        df_ks['Skor_Piotroski_Fuzzy'] = 0.0

    df_ks = df_ks.rename(columns={
        'Current EPS (TTM)':           'Current_EPS_TTM',
        'Current Book Value Per Share': 'Book_Value_Per_Share',
        'Piotroski F-Score':            'Piotroski_F_Score',
    })

    df_notebook = df_ks[[
        'Ticker', 'Current_EPS_TTM', 'Book_Value_Per_Share',
        'Harga_Wajar_Graham', 'Piotroski_F_Score', 'Skor_Piotroski_Fuzzy'
    ]]

    # ── Gabung ──
    df_final = pd.merge(df_a, df_notebook, on='Ticker', how='inner')
    return df_final


@st.cache_data(ttl=3600)
def load_fundamental_data() -> pd.DataFrame:
    """
    Single source of truth — DataFrame gabungan analysis + notebook pipeline.
    Selalu punya semua kolom di _FUNDAMENTAL_DEFAULTS + 'Ticker'.
    Fallback: DataFrame kosong berkolom lengkap (app tidak crash, tapi juga
    tidak menampilkan angka fiktif ke user).
    """
    if not os.path.exists(_EXCEL_PATH):
        print(f"[WARN] Excel tidak ditemukan: {_EXCEL_PATH}")
        return pd.DataFrame(columns=["Ticker"] + list(_FUNDAMENTAL_DEFAULTS.keys()))

    try:
        df = _load_from_excel()
        # Pastikan semua kolom default ada
        for col, default in _FUNDAMENTAL_DEFAULTS.items():
            if col not in df.columns:
                df[col] = default
        print(f"[OK] Fundamental loaded — {len(df)} ticker, {len(df.columns)} kolom")
        return df
    except Exception as e:
        print(f"[ERROR] Gagal load fundamental dari Excel: {e}")
        return pd.DataFrame(columns=["Ticker"] + list(_FUNDAMENTAL_DEFAULTS.keys()))


def get_fundamental_row(ticker: str) -> dict:
    """
    Kembalikan satu baris fundamental sebagai dict yang aman untuk template
    Jinja2 maupun business logic. TIDAK PERNAH mengembalikan DataFrame/Series.

    Dua sumber nilai dalam satu dict:
      • Composite_Rank        (0–1)   → calculate_risk_level, detect_overhyped_status
      • Skor_Piotroski_Fuzzy  (-1/+1) → eksekusi_fuzzy_mamdani

    Jika ticker tidak ditemukan, kembalikan dict default (semua 0.0).
    Nilai default TIDAK ditampilkan ke user — caller wajib cek dan tampilkan 'N/A'.
    """
    df = load_fundamental_data()

    if df.empty or ticker not in df['Ticker'].values:
        print(f"[WARN] Ticker '{ticker}' tidak ada di data fundamental. Pakai default 0.0.")
        result = dict(_FUNDAMENTAL_DEFAULTS)
        result['Ticker'] = ticker
        return result

    row = df[df['Ticker'] == ticker].iloc[0].to_dict()

    # Pastikan semua key default ada dan nilainya bukan NaN
    for col, default in _FUNDAMENTAL_DEFAULTS.items():
        if col not in row or (isinstance(row[col], float) and pd.isna(row[col])):
            row[col] = default

    return row


# ==============================================================================
# STOCK & GOLD DATA
# ==============================================================================

@st.cache_data(ttl=900)
def fetch_stock_data(ticker_code, period="2y"):
    data = yf.download(
        ticker_code, period=period, interval="1d",
        progress=False, auto_adjust=False
    )
    if data.empty:
        raise ValueError("Data harga saham terbaru gagal diambil.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns.name = None
    data = data[["Date", "Open", "High", "Low", "Close", "Volume"]].copy()
    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)
    data["Return"]     = data["Close"].pct_change(fill_method=None)
    data["MA7"]        = data["Close"].rolling(window=7).mean()
    data["MA30"]       = data["Close"].rolling(window=30).mean()
    data["Volatility"] = data["Return"].rolling(window=7).std()
    return data


@st.cache_data(ttl=900)
def fetch_gold_data(period="2y"):
    gold = yf.download(
        "GC=F", period=period, interval="1d",
        progress=False, auto_adjust=False
    )
    if gold.empty:
        raise ValueError("Data harga emas tidak tersedia.")

    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.get_level_values(0)

    gold = gold.reset_index()
    gold.columns.name = None
    gold = gold[["Date", "Close"]].copy()
    gold = gold.rename(columns={"Close": "Gold_Close"})
    gold["Date"] = pd.to_datetime(gold["Date"])
    gold["Gold_Return"] = gold["Gold_Close"].pct_change(fill_method=None)
    return gold


@st.cache_data(ttl=900)
def fetch_news(ticker):
    feedparser.USER_AGENT = "Mozilla/5.0"
    news_items = []
    for keyword in NEWS_KEYWORDS.get(ticker, []):
        query = quote(keyword)
        url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        time.sleep(1)
        for entry in feed.entries[:15]:
            judul = entry.title if "title" in entry else ""
            judul = judul.rsplit(" - ", 1)[0] if " - " in judul else judul
            news_items.append({
                "Ticker": ticker, "Keyword": keyword, "Title": judul,
                "Published": entry.published if "published" in entry else None,
                "Source": entry.source.title if "source" in entry else "Google News",
                "Link": entry.link if "link" in entry else ""
            })

    news_df = pd.DataFrame(news_items)
    if news_df.empty:
        return pd.DataFrame(columns=["Date", "Tanggal", "Ticker", "Title", "Source", "Link"])

    news_df["Published"] = pd.to_datetime(news_df["Published"], errors="coerce")
    news_df["Tanggal"]   = news_df["Published"].dt.date
    news_df["Date"]      = news_df["Published"].dt.normalize()
    news_df["Date"]      = pd.to_datetime(news_df["Date"], errors="coerce")
    news_df["Date"]      = news_df["Date"].replace({pd.NaT: None})
    news_df = news_df.dropna(subset=["Title"]).drop_duplicates(subset=["Title"])
    news_df = news_df[["Date", "Tanggal", "Ticker", "Title", "Source", "Link"]]
    return news_df.sort_values("Date", ascending=False).reset_index(drop=True)


@st.cache_data(ttl=1800)
def fetch_market_news(category="Semua Berita"):
    news_items = []
    if category == "Semua Berita":
        keywords = [kw for kws in MARKET_NEWS_CATEGORIES.values() for kw in kws]
    else:
        keywords = MARKET_NEWS_CATEGORIES.get(category, [])

    for keyword in keywords:
        q = quote(f"{keyword} site:kontan.co.id OR site:cnbcindonesia.com OR site:bisnis.com")
        feed = feedparser.parse(f"https://news.google.com/rss/search?q={q}&hl=id&gl=ID&ceid=ID:id")
        limit = 5 if category == "Semua Berita" else 15
        for entry in feed.entries[:limit]:
            judul = entry.title if "title" in entry else ""
            judul = judul.rsplit(" - ", 1)[0] if " - " in judul else judul
            news_items.append({
                "Keyword": keyword,
                "Category": category if category != "Semua Berita" else next(
                    (k for k, v in MARKET_NEWS_CATEGORIES.items() if keyword in v), "Umum"
                ),
                "Title": judul,
                "Published": entry.published if "published" in entry else None,
                "Source": entry.source.title if "source" in entry else "Google News",
                "Link": entry.link if "link" in entry else ""
            })

    news_df = pd.DataFrame(news_items)
    if news_df.empty:
        return pd.DataFrame(columns=["Date", "Category", "Title", "Source", "Link"])

    news_df["Published"] = pd.to_datetime(news_df["Published"], errors="coerce")
    news_df["Date"]      = pd.to_datetime(news_df["Published"].dt.date, errors="coerce")
    news_df = news_df.dropna(subset=["Title"]).drop_duplicates(subset=["Title"])
    news_df = news_df[["Date", "Category", "Title", "Source", "Link"]]
    return news_df.sort_values("Date", ascending=False).reset_index(drop=True).head(50)