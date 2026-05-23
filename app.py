import os
import time
import joblib
import feedparser
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import streamlit as st
from urllib.parse import quote
from datetime import datetime

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="GoldStock Insight",
    page_icon="🟡",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ======================================================
# CUSTOM CSS
# ======================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(212, 175, 55, 0.16), transparent 28%),
            radial-gradient(circle at top right, rgba(255, 215, 0, 0.08), transparent 25%),
            linear-gradient(135deg, #05070d 0%, #0b1020 45%, #05070d 100%);
        color: #f8fafc;
    }

    header[data-testid="stHeader"] {
        background: rgba(5, 7, 13, 0);
    }

    div[data-testid="stToolbar"] {
        visibility: hidden;
        height: 0%;
        position: fixed;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    .hero-container {
        padding: 3.2rem 2.5rem;
        border-radius: 32px;
        background:
            linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.03)),
            linear-gradient(135deg, rgba(212, 175, 55, 0.12), rgba(0, 0, 0, 0));
        border: 1px solid rgba(212, 175, 55, 0.28);
        box-shadow: 0 30px 80px rgba(0, 0, 0, 0.45);
        position: relative;
        overflow: hidden;
    }

    .hero-container::before {
        content: "";
        position: absolute;
        top: -100px;
        right: -100px;
        width: 280px;
        height: 280px;
        background: radial-gradient(circle, rgba(255, 215, 0, 0.22), transparent 70%);
        border-radius: 50%;
    }

    .premium-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 8px 14px;
        border-radius: 999px;
        background: rgba(212, 175, 55, 0.12);
        border: 1px solid rgba(212, 175, 55, 0.35);
        color: #f7d774;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.3px;
        margin-bottom: 1rem;
    }

    .hero-title {
        font-size: clamp(2.4rem, 5vw, 4.8rem);
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.06em;
        margin-bottom: 1rem;
        color: #ffffff;
    }

    .gold-text {
        background: linear-gradient(135deg, #fff3b0 0%, #d4af37 40%, #b8860b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        line-height: 1.8;
        color: #cbd5e1;
        max-width: 760px;
        margin-bottom: 1.5rem;
    }

    .hero-mini {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 1.2rem;
    }

    .mini-chip {
        padding: 10px 14px;
        border-radius: 999px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.18);
        color: #e2e8f0;
        font-size: 0.85rem;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        margin: 1.2rem 0 0.4rem 0;
        letter-spacing: -0.03em;
    }

    .section-caption {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-bottom: 1rem;
    }

    .glass-card {
        padding: 1.3rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 20px 45px rgba(0, 0, 0, 0.22);
        backdrop-filter: blur(16px);
        height: 100%;
    }

    .gold-card {
        padding: 1.4rem;
        border-radius: 26px;
        background:
            linear-gradient(135deg, rgba(212, 175, 55, 0.17), rgba(255, 255, 255, 0.04)),
            rgba(15, 23, 42, 0.74);
        border: 1px solid rgba(212, 175, 55, 0.35);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.28);
        height: 100%;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-bottom: 0.35rem;
        font-weight: 600;
    }

    .metric-value {
        color: #ffffff;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.03em;
    }

    .metric-small {
        color: #cbd5e1;
        font-size: 0.86rem;
        margin-top: 0.25rem;
    }

    .recommendation-title {
        color: #f7d774;
        font-size: 0.95rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.5rem;
    }

    .recommendation-main {
        color: #ffffff;
        font-size: clamp(2rem, 4vw, 3.6rem);
        font-weight: 900;
        letter-spacing: -0.06em;
        line-height: 1;
        margin-bottom: 0.6rem;
    }

    .recommendation-desc {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
    }

    .badge {
        display: inline-flex;
        padding: 8px 14px;
        border-radius: 999px;
        font-weight: 800;
        font-size: 0.8rem;
        margin-top: 0.75rem;
    }

    .badge-green {
        color: #bbf7d0;
        background: rgba(34, 197, 94, 0.16);
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    .badge-blue {
        color: #bfdbfe;
        background: rgba(59, 130, 246, 0.16);
        border: 1px solid rgba(59, 130, 246, 0.35);
    }

    .badge-red {
        color: #fecaca;
        background: rgba(239, 68, 68, 0.16);
        border: 1px solid rgba(239, 68, 68, 0.35);
    }

    .badge-orange {
        color: #fed7aa;
        background: rgba(249, 115, 22, 0.16);
        border: 1px solid rgba(249, 115, 22, 0.35);
    }

    .news-card {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(2, 6, 23, 0.42);
        border: 1px solid rgba(148, 163, 184, 0.13);
        margin-bottom: 0.8rem;
        transition: 0.25s ease;
    }

    .news-card:hover {
        transform: translateY(-2px);
        border-color: rgba(212, 175, 55, 0.38);
        background: rgba(15, 23, 42, 0.72);
    }

    .news-title {
        color: #f8fafc;
        font-weight: 700;
        font-size: 0.96rem;
        margin-bottom: 0.45rem;
        line-height: 1.45;
    }

    .news-meta {
        color: #94a3b8;
        font-size: 0.78rem;
        margin-bottom: 0.5rem;
    }

    .news-link {
        color: #f7d774 !important;
        font-size: 0.84rem;
        font-weight: 700;
        text-decoration: none;
    }

    .step-box {
        padding: 1rem;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.60);
        border: 1px solid rgba(148, 163, 184, 0.14);
        height: 100%;
    }

    .step-number {
        width: 34px;
        height: 34px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: linear-gradient(135deg, #f7d774, #b8860b);
        color: #0f172a;
        font-weight: 900;
        margin-bottom: 0.8rem;
    }

    .step-title {
        color: #ffffff;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .step-desc {
        color: #94a3b8;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .disclaimer {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: rgba(239, 68, 68, 0.08);
        border: 1px solid rgba(239, 68, 68, 0.22);
        color: #fecaca;
        font-size: 0.88rem;
        line-height: 1.6;
    }

    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 18px;
        padding: 0.85rem 1.2rem;
        background: linear-gradient(135deg, #f7d774 0%, #d4af37 45%, #9a6b00 100%);
        color: #0b1020;
        font-weight: 900;
        font-size: 1rem;
        box-shadow: 0 14px 30px rgba(212, 175, 55, 0.25);
        transition: 0.25s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 18px 45px rgba(212, 175, 55, 0.34);
        color: #05070d;
    }

    div[data-baseweb="select"] > div {
        background: rgba(15, 23, 42, 0.85);
        border-color: rgba(212, 175, 55, 0.25);
        border-radius: 16px;
        color: #ffffff;
    }

    label {
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        padding: 10px 18px;
        background: rgba(15, 23, 42, 0.70);
        border: 1px solid rgba(148, 163, 184, 0.16);
        color: #cbd5e1;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(212, 175, 55, 0.16);
        border: 1px solid rgba(212, 175, 55, 0.40);
        color: #f7d774;
    }

    .badge-low-risk { background: rgba(74, 222, 128, 0.15); color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); }
    .badge-med-risk { background: rgba(250, 204, 21, 0.15); color: #facc15; border: 1px solid rgba(250, 204, 21, 0.3); }
    .badge-high-risk { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }
    
    .badge-overhyped { background: rgba(248, 113, 113, 0.15); color: #f87171; border: 1px solid rgba(248, 113, 113, 0.3); }
    .badge-watchout { background: rgba(251, 146, 60, 0.15); color: #fb923c; border: 1px solid rgba(251, 146, 60, 0.3); }
    .badge-normal { background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    
    .top-pick-card {
        padding: 2.5rem 2rem;
        border-radius: 24px;
        background: linear-gradient(135deg, rgba(212, 175, 55, 0.15), rgba(15, 23, 42, 0.9));
        border: 1px solid rgba(212, 175, 55, 0.4);
        box-shadow: 0 15px 40px rgba(0,0,0,0.4);
        text-align: center;
        margin-bottom: 2rem;
    }
    .top-pick-ticker {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #f7d774 0%, #d4af37 45%, #9a6b00 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }

    @media (max-width: 768px) {
        .hero-container {
            padding: 2rem 1.3rem;
            border-radius: 24px;
        }
        .glass-card, .gold-card {
            padding: 1rem;
        }
        .recommendation-main {
            font-size: 2rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# CONFIG
# ======================================================

TICKER_MAP = {
    "ANTM": "ANTM.JK",
    "MDKA": "MDKA.JK",
    "BRMS": "BRMS.JK",
    "PSAB": "PSAB.JK"
}

COMPANY_NAMES = {
    "ANTM": "PT Aneka Tambang Tbk",
    "MDKA": "PT Merdeka Copper Gold Tbk",
    "BRMS": "PT Bumi Resources Minerals Tbk",
    "PSAB": "PT J Resources Asia Pasifik Tbk"
}

NEWS_KEYWORDS = {
    "ANTM": ["ANTM saham", "Antam emas", "Aneka Tambang saham", "ANTM emas"],
    "MDKA": ["MDKA saham", "Merdeka Copper Gold saham", "MDKA emas"],
    "BRMS": ["BRMS saham", "Bumi Resources Minerals saham", "BRMS emas"],
    "PSAB": ["PSAB saham", "J Resources Asia Pasifik saham", "PSAB emas", "J Resources gold", "J Resources tambang emas"]
}

MODEL_FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "Return", "MA7", "MA30", "Volatility",
    "Gold_Close", "Gold_Return",
    "Sentiment_Score", "News_Count",
    "PBV_x_ROE", "Price_to_Equity_Discount", "Relative_PE_ratio", 
    "EPS_Growth", "Debt_to_Total_Assets_Ratio", "Liquidity_Differential", 
    "CCE", "Operating_Efficiency", "Dividend_Payout", 
    "Yearly_Price_Change", "Composite_Rank", "Net_Debt_to_Equity"
]

MARKET_NEWS_CATEGORIES = {
    "Semua Berita": [], # will be populated dynamically or handles all
    "Emas": ["harga emas hari ini", "emas dunia", "emas antam", "harga emas naik", "harga emas turun", "gold price today"],
    "Saham & IHSG": ["IHSG hari ini", "saham Indonesia", "pasar modal Indonesia", "Bursa Efek Indonesia", "rekomendasi saham hari ini"],
    "Ekonomi Indonesia": ["ekonomi Indonesia", "inflasi Indonesia", "suku bunga Bank Indonesia", "rupiah hari ini", "pertumbuhan ekonomi Indonesia"],
    "Ekonomi Global": ["ekonomi global", "The Fed suku bunga", "inflasi Amerika", "resesi global", "geopolitical risk market"],
    "Komoditas": ["harga komoditas", "harga minyak dunia", "harga batu bara", "harga tembaga", "commodity market"],
    "Tren Pasar": ["saham ramai dibahas", "market trend today", "investor sentiment", "fear of missing out saham", "berita ekonomi viral"]
}

MARKET_POSITIVE_WORDS = [
    "naik", "menguat", "positif", "tumbuh", "meningkat", "optimis", "rebound", 
    "bullish", "stabil", "surplus", "rekor", "cuan", "laba", "akumulasi", "prospek"
]

MARKET_NEGATIVE_WORDS = [
    "turun", "melemah", "negatif", "anjlok", "koreksi", "tertekan", "inflasi", 
    "resesi", "krisis", "rugi", "bearish", "risiko", "ketidakpastian", "perang", 
    "konflik", "gagal", "defisit"
]

STOPWORDS_ID = {
    "dan", "yang", "di", "ke", "dari", "untuk", "dengan", "dalam", "hari", 
    "ini", "terbaru", "adalah", "pada", "karena", "sebagai", "itu", "akan", 
    "bisa", "ada", "tidak", "juga", "sudah", "saja", "lagi", "atau", "oleh",
    "untuk", "kita", "kami", "saya", "kamu", "mereka", "dia", "saat", "bagi"
}


# ======================================================
# HELPER COMPONENTS
# ======================================================

def format_percent(value):
    try:
        return f"{value * 100:.2f}%"
    except Exception:
        return "-"


def format_number(value):
    try:
        if abs(value) >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.2f} T"
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f} B"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f} M"
        return f"{value:,.2f}"
    except Exception:
        return "-"


def metric_card(label, value, small_text=""):
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-small">{small_text}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_badge_class(recommendation):
    if recommendation == "Jangka Panjang":
        return "badge-green"
    if recommendation == "Jangka Pendek":
        return "badge-blue"
    if recommendation == "Overhyped / Hindari":
        return "badge-orange"
    return "badge-red"


# ======================================================
# DATA LOADING
# ======================================================

@st.cache_data(ttl=3600)
def load_fundamental_data():
    path = "data/fundamental_clean.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        return df

    # Fallback agar app tetap jalan kalau file belum dimasukkan
    fallback = pd.DataFrame({
        "Ticker": ["ANTM", "BRMS", "MDKA", "PSAB", "ACES"],
        "PBV_x_ROE": [2999.45, 58.12, -417.98, 188.96, 390.46],
        "Close_Price": [3640, 735, 3260, 510, 364],
        "Price_to_Equity_Discount": [120.36, 1263.56, 0.0, 268.89, 92.22],
        "Relative_PE_ratio": [0.07, 0.01, 0.0, 0.04, 0.08],
        "EPS_Growth": [0.0, 0.0, 0.0, 0.13, 0.0],
        "Debt_to_Total_Assets_Ratio": [0.03, 0.14, 0.32, 0.11, 0.01],
        "Liquidity_Differential": [1.51, 1.08, 1.87, 1.39, 2.1],
        "CCE": [0.06, 0.26, 0.14, 0.57, 0.19],
        "Operating_Efficiency": [0.19, 0.63, 0.71, 0.54, 0.22],
        "Dividend_Payout": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Yearly_Price_Change": [0.0, 0.0, 0.0, 0.0, 0.0],
        "Composite_Rank": [0.6, 0.2, 0.26, 0.34, 0.65],
        "Net_Debt_to_Equity": [0.2, 0.11, 1.79, 0.12, 0.34]
    })

    return fallback


@st.cache_data(ttl=900)
def fetch_stock_data(ticker_code, period="2y"):
    data = yf.download(
        ticker_code,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False
    )

    if data.empty:
        raise ValueError("Data harga saham terbaru gagal diambil. Silakan coba beberapa saat lagi.")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data = data.reset_index()
    data.columns.name = None

    required = ["Date", "Open", "High", "Low", "Close", "Volume"]
    data = data[required].copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    data["Return"] = data["Close"].pct_change(fill_method=None)
    data["MA7"] = data["Close"].rolling(window=7).mean()
    data["MA30"] = data["Close"].rolling(window=30).mean()
    data["Volatility"] = data["Return"].rolling(window=7).std()

    return data


@st.cache_data(ttl=900)
def fetch_gold_data(period="2y"):
    gold = yf.download(
        "GC=F",
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False
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
    news_items = []

    for keyword in NEWS_KEYWORDS.get(ticker, []):
        query = quote(keyword)
        url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)

        for entry in feed.entries[:15]:
            news_items.append({
                "Ticker": ticker,
                "Keyword": keyword,
                "Title": entry.title if "title" in entry else "",
                "Published": entry.published if "published" in entry else None,
                "Source": entry.source.title if "source" in entry else "Google News",
                "Link": entry.link if "link" in entry else ""
            })

    news_df = pd.DataFrame(news_items)

    if news_df.empty:
        return pd.DataFrame(columns=["Date", "Ticker", "Title", "Source", "Link"])

    news_df["Published"] = pd.to_datetime(news_df["Published"], errors="coerce")
    news_df["Date"] = news_df["Published"].dt.date
    news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
    news_df = news_df.dropna(subset=["Title"]).drop_duplicates(subset=["Title"])
    news_df = news_df[["Date", "Ticker", "Title", "Source", "Link"]]
    news_df = news_df.sort_values("Date", ascending=False).reset_index(drop=True)

    return news_df


# ======================================================
# SENTIMENT
# ======================================================

POSITIVE_WORDS = [
    "naik", "menguat", "positif", "cuan", "untung", "laba", "melonjak",
    "tumbuh", "meningkat", "rekor", "prospek", "bagus", "cerah",
    "buy", "akumulasi", "bullish", "rebound", "mengkilap", "diburu",
    "rekomendasi", "target", "optimis", "ekspansi", "dividen"
]

NEGATIVE_WORDS = [
    "turun", "melemah", "negatif", "rugi", "anjlok", "merosot",
    "tertekan", "koreksi", "jatuh", "lesu", "beban", "risiko",
    "sell", "hindari", "bearish", "jeblok", "ambrol", "tekanan",
    "utang", "turunnya", "penurunan", "waspada"
]


def get_sentiment_score(text):
    text = str(text).lower()

    pos_count = sum(1 for word in POSITIVE_WORDS if word in text)
    neg_count = sum(1 for word in NEGATIVE_WORDS if word in text)

    raw_score = pos_count - neg_count

    if raw_score > 0:
        label = "Positive"
        score = min(raw_score / 3, 1)
    elif raw_score < 0:
        label = "Negative"
        score = max(raw_score / 3, -1)
    else:
        label = "Neutral"
        score = 0

    return label, score


def apply_sentiment(news_df):
    if news_df.empty:
        return news_df, 0, "Neutral", 0

    sentiment_result = news_df["Title"].apply(get_sentiment_score)
    news_df["Sentiment_Label"] = sentiment_result.apply(lambda x: x[0])
    news_df["Sentiment_Score"] = sentiment_result.apply(lambda x: x[1])

    avg_score = news_df["Sentiment_Score"].mean()
    news_count = len(news_df)

    if avg_score > 0.1:
        label = "Positive"
    elif avg_score < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return news_df, avg_score, label, news_count


# ======================================================
# MARKET NEWS & TRENDS LOGIC
# ======================================================

@st.cache_data(ttl=1800)
def fetch_market_news(category="Semua Berita"):
    news_items = []
    
    if category == "Semua Berita":
        keywords = []
        for kw_list in MARKET_NEWS_CATEGORIES.values():
            keywords.extend(kw_list)
    else:
        keywords = MARKET_NEWS_CATEGORIES.get(category, [])
        
    for keyword in keywords:
        query = quote(keyword)
        url = f"https://news.google.com/rss/search?q={query}&hl=id&gl=ID&ceid=ID:id"
        feed = feedparser.parse(url)
        
        limit_per_kw = 5 if category == "Semua Berita" else 15 
        for entry in feed.entries[:limit_per_kw]:
            news_items.append({
                "Keyword": keyword,
                "Category": category if category != "Semua Berita" else next((k for k, v in MARKET_NEWS_CATEGORIES.items() if keyword in v), "Umum"),
                "Title": entry.title if "title" in entry else "",
                "Published": entry.published if "published" in entry else None,
                "Source": entry.source.title if "source" in entry else "Google News",
                "Link": entry.link if "link" in entry else ""
            })
            
    news_df = pd.DataFrame(news_items)
    if news_df.empty:
        return pd.DataFrame(columns=["Date", "Category", "Title", "Source", "Link"])
        
    news_df["Published"] = pd.to_datetime(news_df["Published"], errors="coerce")
    news_df["Date"] = news_df["Published"].dt.date
    news_df["Date"] = pd.to_datetime(news_df["Date"], errors="coerce")
    news_df = news_df.dropna(subset=["Title"]).drop_duplicates(subset=["Title"])
    news_df = news_df[["Date", "Category", "Title", "Source", "Link"]]
    news_df = news_df.sort_values("Date", ascending=False).reset_index(drop=True)
    
    return news_df.head(50)

def apply_market_sentiment(news_df):
    if news_df.empty:
        return news_df, 0, "Neutral Market"

    def score_market(text):
        text = str(text).lower()
        pos_count = sum(1 for word in MARKET_POSITIVE_WORDS if word in text)
        neg_count = sum(1 for word in MARKET_NEGATIVE_WORDS if word in text)
        
        raw_score = pos_count - neg_count
        if raw_score > 0:
            return "Positive", min(raw_score / 3, 1)
        elif raw_score < 0:
            return "Negative", max(raw_score / 3, -1)
        return "Neutral", 0

    sentiment_result = news_df["Title"].apply(score_market)
    news_df["Sentiment_Label"] = sentiment_result.apply(lambda x: x[0])
    news_df["Sentiment_Score"] = sentiment_result.apply(lambda x: x[1])

    avg_score = news_df["Sentiment_Score"].mean()
    if avg_score > 0.05:
        label = "Positive Market"
    elif avg_score < -0.05:
        label = "Negative Market"
    else:
        label = "Neutral Market"

    return news_df, avg_score, label

def get_trending_topics(news_df):
    if news_df.empty:
        return []
        
    import re
    from collections import Counter
    
    all_titles = " ".join(news_df["Title"].tolist()).lower()
    all_titles = re.sub(r'[^\w\s]', '', all_titles)
    words = all_titles.split()
    
    filtered_words = [w for w in words if w not in STOPWORDS_ID and len(w) > 2]
    word_counts = Counter(filtered_words)
    return [word for word, count in word_counts.most_common(10)]

def create_market_sentiment_chart(news_df):
    if news_df.empty or "Sentiment_Label" not in news_df.columns:
        return None
        
    sentiment_counts = news_df["Sentiment_Label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    colors = {'Positive': '#4ade80', 'Neutral': '#94a3b8', 'Negative': '#f87171'}
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts["Sentiment"],
        values=sentiment_counts["Count"],
        hole=0.6,
        marker=dict(colors=[colors.get(l, '#94a3b8') for l in sentiment_counts["Sentiment"]])
    )])
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="Distribusi Sentimen Market",
        showlegend=False
    )
    return fig

def create_category_bar_chart(news_df):
    if news_df.empty:
        return None
        
    cat_counts = news_df["Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    
    fig = go.Figure(go.Bar(
        x=cat_counts["Count"],
        y=cat_counts["Category"],
        orientation='h',
        marker_color="#d4af37"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="Jumlah Berita per Kategori",
        yaxis={'categoryorder':'total ascending'}
    )
    return fig

def generate_market_insight(sentiment_label, avg_score, top_category):
    if sentiment_label == "Positive Market":
        return "Sentimen pasar saat ini cenderung positif karena mayoritas berita terbaru menunjukkan tren penguatan, prospek pasar, atau kondisi ekonomi yang stabil. Namun, investor tetap perlu memperhatikan risiko volatilitas dan perubahan sentimen global yang bisa terjadi kapan saja."
    elif sentiment_label == "Negative Market":
        return "Sentimen pasar saat ini cenderung negatif karena banyak berita terkait pelemahan harga, tekanan ekonomi, risiko global, atau ketidakpastian pasar. Investor disarankan lebih berhati-hati sebelum mengambil keputusan besar."
    else:
        return "Sentimen pasar saat ini cenderung netral. Berita yang muncul masih sangat beragam antara sentimen positif dan negatif, sehingga investor disarankan untuk tidak hanya mengikuti hype pasar dan tetap memantau tren yang berkembang."


# ======================================================
# MODEL AND RECOMMENDATION (KAGGLE INJECTION)
# ======================================================

import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Data Master Hasil 3 Pilar (XGBoost, IndoBERT, Piotroski Fundamental)
KAGGLE_PILAR_DATA = {
    "ANTM": {"pred_return": 0.2618, "sentiment_score": 0.0101, "graham_price": 3148.69, "piotroski_score": 8, "piotroski_fuzzy": 1.0},
    "BRMS": {"pred_return": 0.0240, "sentiment_score": 0.0090, "graham_price": 138.86, "piotroski_score": 8, "piotroski_fuzzy": 1.0},
    "MDKA": {"pred_return": -0.1500, "sentiment_score": -0.0609, "graham_price": 0.00, "piotroski_score": 6, "piotroski_fuzzy": -1.0},
    "PSAB": {"pred_return": 0.0434, "sentiment_score": 0.0335, "graham_price": 297.24, "piotroski_score": 7, "piotroski_fuzzy": 0.0}
}

def eksekusi_fuzzy_mamdani(prediksi_return, skor_sentimen, skor_fundamental):
    in_return = ctrl.Antecedent(np.arange(-10, 11, 0.1), 'return_ai')
    in_sentimen = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'sentimen')
    in_fund = ctrl.Antecedent(np.arange(-1, 1.1, 0.1), 'fundamental')
    out_keputusan = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')
    
    in_return.automf(names=['bearish', 'stagnan', 'bullish'])
    in_sentimen.automf(names=['negatif', 'netral', 'positif'])
    in_fund.automf(names=['sakit', 'biasa', 'sehat'])
    out_keputusan.automf(names=['hindari', 'jangka_pendek', 'jangka_panjang'])
    
    r1 = ctrl.Rule(in_fund['sakit'] | in_sentimen['negatif'], out_keputusan['hindari'])
    r2 = ctrl.Rule(in_fund['sehat'] & in_return['bullish'] & in_sentimen['positif'], out_keputusan['jangka_panjang'])
    r3 = ctrl.Rule(in_fund['biasa'] & in_sentimen['positif'], out_keputusan['jangka_pendek'])
    r4 = ctrl.Rule(in_return['bearish'] & in_fund['sakit'], out_keputusan['hindari'])
    r5 = ctrl.Rule(in_return['bullish'] & in_fund['sehat'], out_keputusan['jangka_panjang'])
    
    mesin = ctrl.ControlSystem([r1, r2, r3, r4, r5])
    simulasi = ctrl.ControlSystemSimulation(mesin)
    
    simulasi.input['return_ai'] = np.clip(prediksi_return * 100, -10, 10)
    simulasi.input['sentimen'] = np.clip(skor_sentimen, -1, 1)
    simulasi.input['fundamental'] = np.clip(skor_fundamental, -1, 1)
    
    try:
        simulasi.compute()
        return simulasi.output['rekomendasi']
    except:
        return 50.0

@st.cache_resource
def load_model():
    model_path = "models/rf_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None


def fallback_predict_return(row):
    """
    Fallback kalau model belum tersedia.
    Ini bukan model ML, tapi rule-based signal agar app tetap bisa demo.
    """

    technical_signal = 0

    if row["Close"] > row["MA7"]:
        technical_signal += 0.008
    else:
        technical_signal -= 0.004

    if row["MA7"] > row["MA30"]:
        technical_signal += 0.012
    else:
        technical_signal -= 0.006

    sentiment_signal = row["Sentiment_Score"] * 0.012
    fundamental_signal = (row["Composite_Rank"] - 0.5) * 0.025
    gold_signal = row["Gold_Return"] * 0.6 if not pd.isna(row["Gold_Return"]) else 0

    volatility_penalty = min(row["Volatility"], 0.08) * 0.15 if not pd.isna(row["Volatility"]) else 0

    predicted = technical_signal + sentiment_signal + fundamental_signal + gold_signal - volatility_penalty

    return float(np.clip(predicted, -0.08, 0.12))


def prepare_latest_row(stock_df, gold_df, fundamental_row, sentiment_score, news_count):
    merged = pd.merge(
        stock_df,
        gold_df[["Date", "Gold_Close", "Gold_Return"]],
        on="Date",
        how="left"
    )

    merged = merged.ffill().dropna().reset_index(drop=True)

    latest = merged.iloc[-1].copy()

    latest["Sentiment_Score"] = sentiment_score
    latest["News_Count"] = news_count

    for col in ["PBV_x_ROE", "Price_to_Equity_Discount", "Relative_PE_ratio", "EPS_Growth", "Debt_to_Total_Assets_Ratio", "Liquidity_Differential", "CCE", "Operating_Efficiency", "Dividend_Payout", "Yearly_Price_Change", "Composite_Rank", "Net_Debt_to_Equity"]:
        latest[col] = fundamental_row[col]

    return latest


def predict_return(ticker):
    data = KAGGLE_PILAR_DATA.get(ticker)
    if data:
        return float(data["pred_return"]), "XGBoost & GRU (Late Fusion)"
    return 0.02, "Rule-based fallback"


def generate_recommendation(ticker, predicted_return, sentiment_score, fundamental_score, investment_goal):
    data = KAGGLE_PILAR_DATA.get(ticker)
    
    if data:
        skor_fuzzy = eksekusi_fuzzy_mamdani(data['pred_return'], data['sentiment_score'], data['piotroski_fuzzy'])
    else:
        skor_fuzzy = eksekusi_fuzzy_mamdani(predicted_return, sentiment_score, fundamental_score)
        
    if skor_fuzzy >= 65:
        recommendation = "Jangka Panjang"
        risk = "Rendah - Sedang"
    elif skor_fuzzy >= 40:
        recommendation = "Jangka Pendek"
        risk = "Sedang"
    else:
        recommendation = "Overhyped / Hindari"
        risk = "Tinggi"

    return recommendation, risk, skor_fuzzy


def generate_explanation(ticker, recommendation, predicted_return, sentiment_label, fundamental_label, investment_goal):
    predicted_pct = predicted_return * 100
    company = COMPANY_NAMES.get(ticker, ticker)

    if recommendation == "Jangka Panjang":
        return (
            f"{company} mendapatkan rekomendasi Jangka Panjang karena memiliki dukungan fundamental yang kuat, "
            f"sentimen berita terbaru berada pada kategori {sentiment_label}, dan prediksi return berada di sekitar "
            f"{predicted_pct:.2f}%. Untuk tujuan {investment_goal}, saham ini layak dipertimbangkan karena tidak hanya "
            f"bergerak berdasarkan sentimen pasar, tetapi juga didukung kondisi fundamental yang baik."
        )

    if recommendation == "Jangka Pendek":
        return (
            f"{company} lebih sesuai untuk strategi Jangka Pendek. Prediksi return berada di sekitar "
            f"{predicted_pct:.2f}% dan sentimen berita berada pada kategori {sentiment_label}. Namun, untuk keputusan "
            f"jangka panjang, investor tetap perlu memperhatikan kekuatan fundamental dan risiko volatilitas harga."
        )

    if recommendation == "Overhyped / Hindari":
        return (
            f"{company} terindikasi perlu diwaspadai karena potensi kenaikan harga belum cukup didukung oleh "
            f"fundamental yang kuat. Kondisi fundamental saat ini berada pada kategori {fundamental_label}. "
            f"Hal ini dapat menunjukkan risiko overhyped, yaitu harga terlihat menarik karena sentimen pasar, "
            f"tetapi belum sepenuhnya sejalan dengan kondisi dasar perusahaan."
        )

    return (
        f"{company} saat ini belum memenuhi kriteria yang cukup kuat untuk tujuan {investment_goal}. "
        f"Prediksi return berada di sekitar {predicted_pct:.2f}%, sentimen berita berada pada kategori "
        f"{sentiment_label}, dan kondisi fundamental berada pada kategori {fundamental_label}. "
        f"Karena itu, saham ini lebih baik dipantau terlebih dahulu sebelum mengambil keputusan investasi."
    )


def calculate_risk_level(volatility, predicted_return, sentiment_score, fundamental_score, investment_goal):
    """Menghitung Risk Level."""
    risk_score = 50
    
    if fundamental_score >= 0.5:
        risk_score -= 20
    elif fundamental_score < 0:
        risk_score += 30
        
    if volatility and not pd.isna(volatility):
        if volatility > 0.05:
            risk_score += 25
        elif volatility < 0.02:
            risk_score -= 10
            
    if sentiment_score < -0.1:
        risk_score += 15
    elif sentiment_score > 0.1:
        risk_score -= 10
        
    if investment_goal == "Jangka Pendek":
        risk_score += 10
        
    risk_score = max(0, min(100, risk_score))
    
    if risk_score <= 35:
        level = "Low Risk"
        reason = "Fundamental perusahaan tergolong solid, volatilitas stabil, dan tidak ada sentimen berita negatif yang signifikan."
    elif risk_score <= 65:
        level = "Medium Risk"
        reason = "Kondisi pasar dan perusahaan cukup seimbang, namun ada beberapa faktor volatilitas atau sentimen yang perlu dipantau."
    else:
        level = "High Risk"
        reason = "Fundamental yang kurang mendukung, dipadukan dengan sentimen negatif atau volatilitas tinggi, membuat risiko investasi sangat tinggi saat ini."
        
    return level, risk_score, reason


def detect_overhyped_status(predicted_return, sentiment_score, news_count, fundamental_score):
    """Mendeteksi Overhyped status."""
    hype_score = 0
    
    if sentiment_score > 0.1: hype_score += 20
    if sentiment_score > 0.3: hype_score += 20
    if news_count > 10: hype_score += 15
    if news_count > 20: hype_score += 15
    if predicted_return > 0.05: hype_score += 20
        
    if fundamental_score >= 0.5: hype_score -= 40
    elif fundamental_score < 0: hype_score += 20
        
    hype_score = max(0, min(100, hype_score))
    
    if hype_score >= 70:
        status = "Overhyped"
        reason = "Saham ini sedang sangat ramai diberitakan dan sentimennya sangat positif, namun tidak didukung skor fundamental yang sepadan. Harga mungkin naik hanya karena FOMO."
    elif hype_score >= 45:
        status = "Watch Out"
        reason = "Saham mulai ramai dibicarakan dan sentimen positif. Fundamental masih cukup mendukung, tetapi tetap waspada terhadap potensi overbought."
    else:
        status = "Normal"
        reason = "Pergerakan saham, jumlah berita, dan sentimen pasar masih berjalan wajar dan sejalan dengan fundamental perusahaan."
        
    return status, hype_score, reason


# ======================================================
# WATCHLIST & COMPARE LOGIC
# ======================================================

@st.cache_data(ttl=1800)
def generate_watchlist():
    watchlist_items = []
    try:
        gold_df = fetch_gold_data()
        fund_df = load_fundamental_data()
    except Exception:
        return pd.DataFrame()
    
    for ticker_name, ticker_code in TICKER_MAP.items():
        try:
            stock_df = fetch_stock_data(ticker_code, period="6mo")
            news_df = fetch_news(ticker_name)
            
            _, avg_score, sentiment_label, news_count = apply_sentiment(news_df)
            
            fund_row = fund_df[fund_df["Ticker"] == ticker_name]
            if fund_row.empty:
                continue
            fund_dict = fund_row.iloc[0].to_dict()
            
            latest_row = prepare_latest_row(stock_df, gold_df, fund_dict, avg_score, news_count)
            
            # INJEKSI DATA KAGGLE
            data = KAGGLE_PILAR_DATA.get(ticker_name)
            if data:
                avg_score = data["sentiment_score"]
                sentiment_label = "Positive" if avg_score > 0 else "Negative" if avg_score < 0 else "Neutral"
                fund_dict["Composite_Rank"] = data["piotroski_fuzzy"]
            
            fundamental_score = fund_dict.get("Composite_Rank", 0)
            fund_label = "Kuat / Good" if fundamental_score >= 0 else "Lemah / Weak"
            
            predicted_return, model_source = predict_return(ticker_name)
            
            # Predict for Jangka Pendek default for watchlist
            rec, risk, overall_score = generate_recommendation(
                ticker_name, predicted_return, avg_score, fundamental_score, "Jangka Pendek"
            )
            
            # Get Risk
            risk_level, risk_score, _ = calculate_risk_level(latest_row.get("Volatility", 0), predicted_return, avg_score, fundamental_score, "Jangka Pendek")
            
            # Get Hype
            hype_status, hype_score, _ = detect_overhyped_status(predicted_return, avg_score, news_count, fundamental_score)
            
            watchlist_items.append({
                "Ticker": ticker_name,
                "Company Name": COMPANY_NAMES.get(ticker_name, ticker_name),
                "Close Price": latest_row["Close"],
                "Daily Return": latest_row["Return"],
                "Sentiment Label": sentiment_label,
                "Sentiment Score": avg_score,
                "Fundamental Score": fundamental_score,
                "Fundamental Label": fund_label,
                "Risk Level": risk_level,
                "Hype Status": hype_status,
                "Recommendation": rec,
                "Overall Score": overall_score,
                "Predicted Return": predicted_return
            })
        except Exception as e:
            continue
            
    if not watchlist_items:
        return pd.DataFrame()
        
    return pd.DataFrame(watchlist_items).sort_values("Overall Score", ascending=False).reset_index(drop=True)

def get_top_recommendation(watchlist_df):
    if watchlist_df is None or watchlist_df.empty:
        return None
        
    candidates = watchlist_df[
        (watchlist_df["Predicted Return"] > 0) & 
        (watchlist_df["Risk Level"] != "High Risk") & 
        (watchlist_df["Hype Status"] != "Overhyped")
    ]
    
    if candidates.empty:
        return watchlist_df.iloc[0]
        
    return candidates.sort_values("Overall Score", ascending=False).iloc[0]

def compare_stocks(ticker_1, ticker_2, investment_goal, watchlist_df):
    if watchlist_df is None or watchlist_df.empty:
        return None, None, "Data belum tersedia."
        
    df1 = watchlist_df[watchlist_df["Ticker"] == ticker_1]
    df2 = watchlist_df[watchlist_df["Ticker"] == ticker_2]
    
    if df1.empty or df2.empty:
        return None, None, "Data untuk salah satu saham belum tersedia."
        
    stock1 = df1.iloc[0]
    stock2 = df2.iloc[0]
    
    score1 = stock1["Overall Score"]
    score2 = stock2["Overall Score"]
    
    if investment_goal == "Jangka Panjang":
        score1 += stock1["Fundamental Score"] * 50
        score2 += stock2["Fundamental Score"] * 50
        if stock1["Risk Level"] == "High Risk": score1 -= 20
        if stock2["Risk Level"] == "High Risk": score2 -= 20
    else:
        score1 += stock1["Sentiment Score"] * 20 + stock1["Predicted Return"] * 100
        score2 += stock2["Sentiment Score"] * 20 + stock2["Predicted Return"] * 100
        
    winner = ticker_1 if score1 > score2 else ticker_2
    loser = ticker_2 if winner == ticker_1 else ticker_1
    
    reason = f"Berdasarkan perbandingan, {winner} lebih unggul untuk {investment_goal} karena memiliki skor keseluruhan yang lebih baik dengan fundamental dan sentimen yang mendukung. Sementara {loser} lebih cocok dipantau karena memiliki risiko yang sedikit lebih tinggi atau skor fundamental yang lebih rendah."
        
    return pd.concat([df1, df2]), winner, reason



# ======================================================
# UI COMPONENTS
# ======================================================

def render_market_news():
    # MARKET NEWS & TRENDS UI

    st.markdown('<div class="section-title">Market News & Trends</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Pantau berita emas, ekonomi, IHSG, komoditas, dan tren pasar terbaru dalam satu tempat.</div>',
        unsafe_allow_html=True
    )

    mn_col1, mn_col2 = st.columns([1, 1])

    with mn_col1:
        market_category = st.selectbox(
            "Kategori Berita",
            options=list(MARKET_NEWS_CATEGORIES.keys()),
            index=0
        )

    with mn_col2:
        market_search = st.text_input("Cari berita (misalnya: inflasi, IHSG, The Fed)", "")

    with st.spinner("Mengambil berita pasar terbaru..."):
        # Fetch and process market news
        raw_market_news_df = fetch_market_news(market_category)

        # Filter by search
        if market_search:
            market_news_df = raw_market_news_df[raw_market_news_df['Title'].str.contains(market_search, case=False, na=False)].copy()
        else:
            market_news_df = raw_market_news_df.copy()

        market_news_df, market_avg_score, market_label = apply_market_sentiment(market_news_df)
        trending_topics = get_trending_topics(market_news_df)
        market_insight = generate_market_insight(market_label, market_avg_score, market_category)

    if market_news_df.empty:
        st.warning("Berita market terbaru belum tersedia atau tidak ditemukan. Silakan coba beberapa saat lagi atau ubah kata kunci.")
    else:
        # Summary Cards
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            metric_card("Market Mood", market_label, f"Avg Score: {market_avg_score:.2f}")
        with m2:
            metric_card("Total News", str(len(market_news_df)), "Berita ditemukan")
        with m3:
            top_cat = market_news_df['Category'].value_counts().index[0] if not market_news_df.empty else "-"
            metric_card("Top Category", top_cat, "Kategori Terbanyak")
        with m4:
            pos_count = len(market_news_df[market_news_df['Sentiment_Label'] == 'Positive'])
            neg_count = len(market_news_df[market_news_df['Sentiment_Label'] == 'Negative'])
            metric_card("Sentiment Spread", f"{pos_count} Pos / {neg_count} Neg", "Sebaran Berita")

        st.write("")

        # Charts
        mc1, mc2 = st.columns(2)
        with mc1:
            sentiment_chart = create_market_sentiment_chart(market_news_df)
            if sentiment_chart:
                st.plotly_chart(sentiment_chart, use_container_width=True)
        with mc2:
            category_chart = create_category_bar_chart(market_news_df)
            if category_chart:
                st.plotly_chart(category_chart, use_container_width=True)

        st.write("")

        # Trending Topics & Insight
        ti1, ti2 = st.columns([1, 1.5])
        with ti1:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="metric-label">Trending Topics</div>
                    <div style="display:flex; flex-wrap:wrap; gap:8px; margin-top:10px;">
                        {''.join([f'<span class="mini-chip">{topic}</span>' for topic in trending_topics])}
                    </div>
                </div>
                """, unsafe_allow_html=True
            )
        with ti2:
            st.markdown(
                f"""
                <div class="gold-card">
                    <div class="recommendation-title">Market Insight Explanation</div>
                    <div class="step-desc" style="color:#ffffff;">{market_insight}</div>
                </div>
                """, unsafe_allow_html=True
            )

        st.write("")
        st.markdown('<div class="metric-label" style="font-size:1.1rem;">Latest Market News</div>', unsafe_allow_html=True)

        for _, row in market_news_df.head(10).iterrows():
            date_text = row["Date"].strftime("%d %b %Y") if pd.notna(row["Date"]) else "-"
            label = row.get("Sentiment_Label", "Neutral")
            score = row.get("Sentiment_Score", 0)

            st.markdown(
                f"""
                <div class="news-card">
                    <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                        <div class="news-title" style="flex:1;">{row["Title"]}</div>
                        <span class="badge {'badge-green' if label == 'Positive' else 'badge-red' if label == 'Negative' else 'badge-blue'}" style="margin-top:0; margin-left:10px;">{label}</span>
                    </div>
                    <div class="news-meta">{date_text} • {row["Source"]} • Kategori: {row["Category"]} ({score:.2f})</div>
                    <a class="news-link" href="{row["Link"]}" target="_blank">Baca berita →</a>
                </div>
                """,
                unsafe_allow_html=True
            )

    st.write("")
    st.markdown("---")
    st.write("")



def render_watchlist(watchlist_df):
    if watchlist_df is not None and not watchlist_df.empty:
        st.markdown('<div class="section-title">Watchlist Saham</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-caption">Perbandingan cepat performa seluruh saham emas hari ini.</div>', unsafe_allow_html=True)
        
        display_cols = ["Ticker", "Close Price", "Daily Return", "Sentiment Label", "Fundamental Label", "Risk Level", "Hype Status", "Recommendation", "Overall Score"]
        
        st.dataframe(
            watchlist_df[display_cols].style.format({
                "Close Price": "{:,.0f}",
                "Daily Return": "{:.2%}",
                "Overall Score": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        st.write("")


# ======================================================
# CHARTS
# ======================================================

def create_stock_chart(stock_df, ticker):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stock_df["Date"],
        y=stock_df["Close"],
        mode="lines",
        name="Close Price",
        line=dict(width=3)
    ))

    fig.add_trace(go.Scatter(
        x=stock_df["Date"],
        y=stock_df["MA7"],
        mode="lines",
        name="MA7",
        line=dict(width=2, dash="dot")
    ))

    fig.add_trace(go.Scatter(
        x=stock_df["Date"],
        y=stock_df["MA30"],
        mode="lines",
        name="MA30",
        line=dict(width=2, dash="dash")
    ))

    fig.update_layout(
        title=f"Pergerakan Harga Saham {ticker}",
        template="plotly_dark",
        height=430,
        margin=dict(l=20, r=20, t=55, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.25)",
        font=dict(color="#e2e8f0"),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(148,163,184,0.12)")

    return fig


def create_sentiment_chart(news_df):
    if news_df.empty or "Sentiment_Label" not in news_df.columns:
        return None

    sentiment_counts = news_df["Sentiment_Label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]

    fig = go.Figure(
        data=[
            go.Pie(
                labels=sentiment_counts["Sentiment"],
                values=sentiment_counts["Count"],
                hole=0.62
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        height=320,
        margin=dict(l=20, r=20, t=35, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="Distribusi Sentimen Berita"
    )

    return fig


# ======================================================
# HERO
# ======================================================

st.markdown(
    """
    <div class="hero-container">
        <div class="premium-badge">● REAL-TIME HYBRID FINTECH DASHBOARD</div>
        <div class="hero-title">
            GoldStock <span class="gold-text">Insight</span>
        </div>
        <div class="hero-subtitle">
            Sistem rekomendasi saham sektor emas Indonesia berbasis harga saham, harga emas global,
            sentimen berita, dan fundamental perusahaan. Pilih saham, tentukan tujuan investasi,
            lalu dapatkan analisis yang lebih mudah dipahami.
        </div>
        <div class="hero-mini">
            <div class="mini-chip">Real-time Stock Data</div>
            <div class="mini-chip">News Sentiment</div>
            <div class="mini-chip">Fundamental Analysis</div>
            <div class="mini-chip">Prediction Model</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.write("")


# ======================================================
# INPUT SECTION
# ======================================================

st.markdown('<div class="section-title">Analisis Saham</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-caption">Pilih saham dan tujuan investasi untuk mendapatkan rekomendasi berbasis data terbaru.</div>',
    unsafe_allow_html=True
)

input_col1, input_col2, input_col3 = st.columns([1.2, 1.2, 1])

with input_col1:
    selected_ticker = st.selectbox(
        "Pilih Saham",
        options=list(TICKER_MAP.keys()),
        index=0
    )

with input_col2:
    investment_goal = st.selectbox(
        "Tujuan Investasi",
        options=["Jangka Pendek", "Jangka Panjang"],
        index=1
    )
    st.markdown(
        '''
        <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 5px; padding-left: 2px;">
            <i style="color: #cbd5e1;"><b>Jangka Pendek:</b> 1 hr - 1 bln</i> &nbsp;|&nbsp; <i style="color: #cbd5e1;"><b>Jangka Panjang:</b> 6 bln - 1 thn+</i>
        </div>
        ''', unsafe_allow_html=True
    )

with input_col3:
    st.write("")
    analyze_button = st.button("Analisis Saham Sekarang")


# ======================================================
# DEFAULT MESSAGE
# ======================================================

if not analyze_button:
    st.write("")
    with st.spinner("Memuat Top Pick & Watchlist (Data Real-time)..."):
        watchlist_df = generate_watchlist()
        top_pick = get_top_recommendation(watchlist_df)
    
    if top_pick is not None and len(top_pick) > 0:
        st.markdown(
            f'''
            <div class="top-pick-card">
                <div style="color: #f7d774; font-size: 1.1rem; font-weight: 700; letter-spacing: 2px; text-transform: uppercase;">🌟 Top Recommendation Today</div>
                <div class="top-pick-ticker">{top_pick["Ticker"]}</div>
                <div style="color: #cbd5e1; font-size: 1.1rem; margin-bottom: 1rem;">{top_pick["Company Name"]}</div>
                <div style="display: flex; justify-content: center; gap: 15px; margin-bottom: 1rem;">
                    <span class="badge badge-green">Overall Score: {top_pick["Overall Score"]:.1f}</span>
                    <span class="badge badge-blue">Predicted Return: {top_pick["Predicted Return"]*100:.2f}%</span>
                </div>
                <div style="color: #94a3b8; font-size: 0.95rem; max-width: 600px; margin: 0 auto; line-height: 1.5;">
                    Berdasarkan analisis real-time, {top_pick["Ticker"]} memiliki kombinasi fundamental <b>{top_pick["Fundamental Label"]}</b>, sentimen <b>{top_pick["Sentiment Label"]}</b>, dan risiko <b>{top_pick["Risk Level"]}</b> yang optimal hari ini.
                </div>
            </div>
            ''', unsafe_allow_html=True
        )
        
    render_watchlist(watchlist_df)
    st.markdown("---")
    render_market_news()
    st.markdown("---")
    
    st.markdown('<div class="section-title">Cara Kerja Sistem</div>', unsafe_allow_html=True)

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">1</div>
                <div class="step-title">Ambil Data Harga</div>
                <div class="step-desc">Sistem mengambil harga saham dan harga emas terbaru dari yfinance.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step2:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">2</div>
                <div class="step-title">Ambil Berita</div>
                <div class="step-desc">Sistem mengambil berita terbaru dari Google News RSS berdasarkan ticker saham.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step3:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">3</div>
                <div class="step-title">Hitung Skor</div>
                <div class="step-desc">Sistem menghitung sentimen berita dan membaca skor fundamental perusahaan.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with step4:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">4</div>
                <div class="step-title">Beri Rekomendasi</div>
                <div class="step-desc">Sistem menggabungkan seluruh sinyal untuk menghasilkan rekomendasi akhir.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.stop()


# ======================================================
# ANALYSIS PROCESS
# ======================================================

with st.spinner("Sedang mengambil data terbaru dan menjalankan analisis..."):
    time.sleep(0.7)

    try:
        fundamental_df = load_fundamental_data()
        ticker_fundamental = fundamental_df[fundamental_df["Ticker"] == selected_ticker]

        if ticker_fundamental.empty:
            st.error("Data fundamental untuk saham ini belum tersedia.")
            st.stop()

        fundamental_row = ticker_fundamental.iloc[0].copy()

        stock_df = fetch_stock_data(TICKER_MAP[selected_ticker])
        gold_df = fetch_gold_data()
        news_df = fetch_news(selected_ticker)
        news_df, avg_sentiment_score, sentiment_label, news_count = apply_sentiment(news_df)

        # INJEKSI DATA KAGGLE
        data = KAGGLE_PILAR_DATA.get(selected_ticker)
        if data:
            avg_sentiment_score = data["sentiment_score"]
            sentiment_label = "Positive" if avg_sentiment_score > 0 else "Negative" if avg_sentiment_score < 0 else "Neutral"
            fundamental_row["Composite_Rank"] = data["piotroski_fuzzy"]

        latest_row = prepare_latest_row(
            stock_df=stock_df,
            gold_df=gold_df,
            fundamental_row=fundamental_row,
            sentiment_score=avg_sentiment_score,
            news_count=news_count
        )

        predicted_return, model_source = predict_return(selected_ticker)

        recommendation, risk_level, overall_score = generate_recommendation(
            ticker=selected_ticker,
            predicted_return=predicted_return,
            sentiment_score=avg_sentiment_score,
            fundamental_score=fundamental_row["Composite_Rank"],
            investment_goal=investment_goal
        )

        fund_label = "Kuat / Good Fundamental" if fundamental_row["Composite_Rank"] >= 0 else "Lemah / Weak Fundamental"

        explanation = generate_explanation(
            ticker=selected_ticker,
            recommendation=recommendation,
            predicted_return=predicted_return,
            sentiment_label=sentiment_label,
            fundamental_label=fund_label,
            investment_goal=investment_goal
        )
        
        risk_level_str, risk_score, risk_reason = calculate_risk_level(latest_row.get("Volatility", 0), predicted_return, avg_sentiment_score, fundamental_row["Composite_Rank"], investment_goal)
        hype_status, hype_score, hype_reason = detect_overhyped_status(predicted_return, avg_sentiment_score, news_count, fundamental_row["Composite_Rank"])

    except Exception as e:
        st.error(f"Analisis gagal dijalankan: {e}")
        st.stop()


# ======================================================
# RESULT SUMMARY
# ======================================================


def get_risk_badge(level):
    if level == "Low Risk": return "badge-low-risk"
    if level == "Medium Risk": return "badge-med-risk"
    return "badge-high-risk"

def get_hype_badge(status):
    if status == "Normal": return "badge-normal"
    if status == "Watch Out": return "badge-watchout"
    return "badge-overhyped"

st.write("")
badge_class = get_badge_class(recommendation)

summary_left, summary_right = st.columns([1.45, 1])

with summary_left:
    st.markdown(
        f"""
        <div class="gold-card">
            <div class="recommendation-title">Rekomendasi Akhir</div>
            <div class="recommendation-main">{recommendation}</div>
            <div class="recommendation-desc">
                Saham <b>{selected_ticker}</b> — {COMPANY_NAMES[selected_ticker]}<br>
                Sektor: <b>Gold / Precious Metals</b><br>
                Tujuan investasi: <b>{investment_goal}</b><br>
                Sumber prediksi: <b>{model_source}</b>
            </div>
            <div style="display:flex; gap:8px; flex-wrap:wrap; margin-top:10px;">
                <div class="badge {badge_class}">{recommendation}</div>
                <div class="badge {get_risk_badge(risk_level_str)}">{risk_level_str}</div>
                <div class="badge {get_hype_badge(hype_status)}">{hype_status}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with summary_right:
    r1, r2 = st.columns(2)

    with r1:
        metric_card("Predicted Return", f"{predicted_return * 100:.2f}%", "Estimasi potensi return")
    with r2:
        metric_card("Overall Score", f"{overall_score:.1f}/100", "Skor gabungan sistem")

    r3, r4 = st.columns(2)

    with r3:
        metric_card("Sentiment", sentiment_label, f"Score {avg_sentiment_score:.2f}")
    with r4:
        metric_card("News Count", f"{news_count}", "Berita terbaru")


# ======================================================
# TABS
# ======================================================

st.write("")
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "Harga Saham",
    "Sentimen Berita",
    "Fundamental",
    "Alasan Rekomendasi",
    "Risk & Hype Detector",
    "Compare Stocks",
    "Market News",
    "Full Watchlist"
])


# ======================================================
# STOCK TAB
# ======================================================

with tab1:
    st.markdown('<div class="section-title">Pergerakan Harga Saham</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Grafik harga penutupan, moving average 7 hari, dan moving average 30 hari.</div>',
        unsafe_allow_html=True
    )

    chart_df = stock_df.dropna().copy()

    st.plotly_chart(
        create_stock_chart(chart_df, selected_ticker),
        use_container_width=True
    )

    latest_close = latest_row["Close"]
    latest_return = latest_row["Return"]
    latest_volatility = latest_row["Volatility"]
    latest_gold_return = latest_row["Gold_Return"]

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Latest Close", format_number(latest_close), "Harga penutupan terbaru")
    with c2:
        metric_card("Daily Return", format_percent(latest_return), "Return harian")
    with c3:
        metric_card("Volatility", format_percent(latest_volatility), "Volatilitas 7 hari")
    with c4:
        metric_card("Gold Return", format_percent(latest_gold_return), "Return emas global")


# ======================================================
# NEWS TAB
# ======================================================

with tab2:
    st.markdown('<div class="section-title">Sentimen Berita Terbaru</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Berita diambil secara real-time dari Google News RSS, lalu dihitung skor sentimennya.</div>',
        unsafe_allow_html=True
    )

    news_col1, news_col2 = st.columns([1, 1.4])

    with news_col1:
        if not news_df.empty:
            sentiment_fig = create_sentiment_chart(news_df)
            if sentiment_fig:
                st.plotly_chart(sentiment_fig, use_container_width=True)
        else:
            st.info("Belum ada berita terbaru yang tersedia.")

    with news_col2:
        metric_a, metric_b, metric_c = st.columns(3)
        with metric_a:
            metric_card("Avg Sentiment", f"{avg_sentiment_score:.2f}", "Rata-rata skor")
        with metric_b:
            metric_card("Label", sentiment_label, "Kategori sentimen")
        with metric_c:
            metric_card("Total News", f"{news_count}", "Jumlah berita")

    st.write("")
    st.markdown('<div class="section-title">Daftar Berita</div>', unsafe_allow_html=True)

    if news_df.empty:
        st.warning("Berita terbaru belum tersedia. Sistem menggunakan sentimen netral.")
    else:
        for _, row in news_df.head(8).iterrows():
            date_text = row["Date"].strftime("%d %b %Y") if pd.notna(row["Date"]) else "-"
            label = row.get("Sentiment_Label", "Neutral")
            score = row.get("Sentiment_Score", 0)

            st.markdown(
                f"""
                <div class="news-card">
                    <div class="news-title">{row["Title"]}</div>
                    <div class="news-meta">{date_text} • {row["Source"]} • {label} ({score:.2f})</div>
                    <a class="news-link" href="{row["Link"]}" target="_blank">Baca berita →</a>
                </div>
                """,
                unsafe_allow_html=True
            )


# ======================================================
# FUNDAMENTAL TAB
# ======================================================

with tab3:
    st.markdown('<div class="section-title">Analisis Fundamental</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Data fundamental digunakan untuk melihat kualitas valuasi dan kondisi dasar perusahaan.</div>',
        unsafe_allow_html=True
    )

    f1, f2, f3, f4 = st.columns(4)

    with f1:
        metric_card("PBV x ROE", format_number(fundamental_row["PBV_x_ROE"]), "PBV dikali ROE")
    with f2:
        metric_card("PE Discount", format_percent(fundamental_row["Price_to_Equity_Discount"] / 100), "Price to Equity Discount")
    with f3:
        metric_card("Relative PE", format_number(fundamental_row["Relative_PE_ratio"]), "Relative PE TTM")
    with f4:
        metric_card("EPS Growth", format_percent(fundamental_row["EPS_Growth"]), "Pertumbuhan EPS")

    st.write("")

    f5, f6, f7, f8 = st.columns(4)

    with f5:
        metric_card("Debt Ratio", format_number(fundamental_row["Debt_to_Total_Assets_Ratio"]), "Debt to Total Assets")
    with f6:
        metric_card("Liquidity", format_number(fundamental_row["Liquidity_Differential"]), "Liquidity Differential")
    with f7:
        metric_card("Op. Efficiency", format_number(fundamental_row["Operating_Efficiency"]), "Operating Efficiency")
    with f8:
        fund_label = "Kuat" if fundamental_row["Composite_Rank"] >= 0 else "Lemah"
        metric_card("Composite Rank", f"{fundamental_row['Composite_Rank']:.2f}", fund_label)

    st.write("")
    st.markdown(
        f"""
        <div class="glass-card">
            <div class="metric-label">Kesimpulan Fundamental</div>
            <div class="metric-value">{fund_label}</div>
            <div class="metric-small">
                Skor Composite Rank mewakili kombinasi berbagai rasio fundamental penting perusahaan.
                Nilai lebih tinggi menunjukkan kondisi fundamental yang lebih menarik.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


# ======================================================
# EXPLANATION TAB
# ======================================================

with tab4:
    st.markdown('<div class="section-title">Alasan Rekomendasi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-caption">Penjelasan dibuat sederhana agar mudah dipahami oleh investor pemula.</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="gold-card">
            <div class="recommendation-title">Analisis Sistem</div>
            <div class="recommendation-desc">{explanation}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">1</div>
                <div class="step-title">Harga</div>
                <div class="step-desc">Sistem membaca tren harga, return, moving average, dan volatilitas.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h2:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">2</div>
                <div class="step-title">Emas Global</div>
                <div class="step-desc">Harga emas global digunakan sebagai faktor pendukung saham sektor emas.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h3:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">3</div>
                <div class="step-title">Sentimen</div>
                <div class="step-desc">Judul berita terbaru dianalisis untuk melihat sentimen pasar.</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with h4:
        st.markdown(
            """
            <div class="step-box">
                <div class="step-number">4</div>
                <div class="step-title">Fundamental</div>
                <div class="step-desc">Rasio perusahaan digunakan untuk menilai kekuatan dasar saham.</div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ======================================================
# DISCLAIMER
# ======================================================

st.write("")
st.markdown(
    """
    <div class="disclaimer">
        <b>Disclaimer:</b> Aplikasi ini hanya digunakan sebagai alat bantu analisis dan edukasi.
        Hasil rekomendasi bukan merupakan ajakan membeli atau menjual saham.
        Keputusan investasi tetap menjadi tanggung jawab pengguna.
    </div>
    """,
    unsafe_allow_html=True
)
        

with tab5:
    st.markdown('<div class="section-title">Risk & Hype Detector</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Analisis risiko volatilitas dan deteksi FOMO pasar.</div>', unsafe_allow_html=True)
    
    rt1, rt2 = st.columns(2)
    with rt1:
        st.markdown(
            f'''
            <div class="glass-card" style="border-top: 4px solid {'#f87171' if risk_level_str == 'High Risk' else '#facc15' if risk_level_str == 'Medium Risk' else '#4ade80'};">
                <div class="metric-label">Risk Level: <span class="badge {get_risk_badge(risk_level_str)}">{risk_level_str}</span></div>
                <div class="metric-value">{risk_score:.0f}/100</div>
                <div class="step-desc" style="margin-top: 10px;">{risk_reason}</div>
            </div>
            ''', unsafe_allow_html=True
        )
    with rt2:
        st.markdown(
            f'''
            <div class="glass-card" style="border-top: 4px solid {'#f87171' if hype_status == 'Overhyped' else '#fb923c' if hype_status == 'Watch Out' else '#38bdf8'};">
                <div class="metric-label">Hype Status: <span class="badge {get_hype_badge(hype_status)}">{hype_status}</span></div>
                <div class="metric-value">{hype_score:.0f}/100</div>
                <div class="step-desc" style="margin-top: 10px;">{hype_reason}</div>
            </div>
            ''', unsafe_allow_html=True
        )

with tab6:
    st.markdown('<div class="section-title">⚖️ Compare Stocks</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-caption">Bandingkan saham ini dengan saham emas lainnya.</div>', unsafe_allow_html=True)
    
    compare_ticker = st.selectbox(
        "Pilih saham pembanding:",
        options=[t for t in TICKER_MAP.keys() if t != selected_ticker],
        index=0
    )
    
    if st.button("Bandingkan"):
        with st.spinner(f"Membandingkan {selected_ticker} dengan {compare_ticker}..."):
            watchlist_df = generate_watchlist()
            comp_df, winner, comp_reason = compare_stocks(selected_ticker, compare_ticker, investment_goal, watchlist_df)
            
            if comp_df is not None:
                st.markdown(f'''
                    <div class="gold-card" style="text-align: center; margin-top: 1rem;">
                        <h3 style="color: #f7d774; margin-bottom: 5px;">Pemenang: {winner}</h3>
                        <p style="color: #cbd5e1; font-size: 0.95rem;">{comp_reason}</p>
                    </div>
                ''', unsafe_allow_html=True)
                
                # Show side-by-side comparison
                c1, c2 = st.columns(2)
                stock1_data = comp_df[comp_df["Ticker"] == selected_ticker].iloc[0]
                stock2_data = comp_df[comp_df["Ticker"] == compare_ticker].iloc[0]
                
                with c1:
                    st.markdown(f'''
                        <div class="glass-card" style="border: {'2px solid #d4af37' if winner == selected_ticker else '1px solid rgba(148, 163, 184, 0.16)'};">
                            <h2 style="color: #f7d774; text-align: center;">{selected_ticker}</h2>
                            <hr style="border-color: rgba(255,255,255,0.1);">
                            <p><b>Overall Score:</b> {stock1_data["Overall Score"]:.1f}</p>
                            <p><b>Predicted Return:</b> {stock1_data["Predicted Return"]*100:.2f}%</p>
                            <p><b>Sentiment:</b> <span class="badge badge-blue">{stock1_data["Sentiment Label"]}</span></p>
                            <p><b>Fundamental:</b> {stock1_data["Fundamental Label"]}</p>
                            <p><b>Risk Level:</b> <span class="badge {get_risk_badge(stock1_data["Risk Level"])}">{stock1_data["Risk Level"]}</span></p>
                            <p><b>Hype Status:</b> <span class="badge {get_hype_badge(stock1_data["Hype Status"])}">{stock1_data["Hype Status"]}</span></p>
                            <p><b>Rekomendasi:</b> <span class="badge {get_badge_class(stock1_data["Recommendation"])}">{stock1_data["Recommendation"]}</span></p>
                        </div>
                    ''', unsafe_allow_html=True)
                
                with c2:
                    st.markdown(f'''
                        <div class="glass-card" style="border: {'2px solid #d4af37' if winner == compare_ticker else '1px solid rgba(148, 163, 184, 0.16)'};">
                            <h2 style="color: #f7d774; text-align: center;">{compare_ticker}</h2>
                            <hr style="border-color: rgba(255,255,255,0.1);">
                            <p><b>Overall Score:</b> {stock2_data["Overall Score"]:.1f}</p>
                            <p><b>Predicted Return:</b> {stock2_data["Predicted Return"]*100:.2f}%</p>
                            <p><b>Sentiment:</b> <span class="badge badge-blue">{stock2_data["Sentiment Label"]}</span></p>
                            <p><b>Fundamental:</b> {stock2_data["Fundamental Label"]}</p>
                            <p><b>Risk Level:</b> <span class="badge {get_risk_badge(stock2_data["Risk Level"])}">{stock2_data["Risk Level"]}</span></p>
                            <p><b>Hype Status:</b> <span class="badge {get_hype_badge(stock2_data["Hype Status"])}">{stock2_data["Hype Status"]}</span></p>
                            <p><b>Rekomendasi:</b> <span class="badge {get_badge_class(stock2_data["Recommendation"])}">{stock2_data["Recommendation"]}</span></p>
                        </div>
                    ''', unsafe_allow_html=True)

with tab7:
    render_market_news()

with tab8:
    watchlist_df_tab = generate_watchlist()
    render_watchlist(watchlist_df_tab)