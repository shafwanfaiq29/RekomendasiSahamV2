import os
import pandas as pd
import yfinance as yf
import feedparser
from urllib.parse import quote
from datetime import datetime
from functools import lru_cache
from config import NEWS_KEYWORDS, MARKET_NEWS_CATEGORIES

class DummyST:
    def cache_data(self, *args, **kwargs):
        return lru_cache(maxsize=32)

st = DummyST()

@st.cache_data(ttl=3600)
def load_fundamental_data():
    path = "data/fundamental_clean.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        return df

    fallback = pd.DataFrame({
        "Ticker": ["ANTM", "BRMS", "MDKA", "PSAB"],
        "PBV_x_ROE": [2999.45, 58.12, -417.98, 188.96],
        "Close_Price": [3640, 735, 3260, 200],
        "Price_to_Equity_Discount": [120.36, 1263.56, 0.0, 268.89],
        "Relative_PE_ratio": [0.07, 0.01, 0.0, 0.04],
        "EPS_Growth": [0.0, 0.0, 0.0, 0.13],
        "Debt_to_Total_Assets_Ratio": [0.03, 0.14, 0.32, 0.11],
        "Liquidity_Differential": [1.51, 1.08, 1.87, 1.39],
        "CCE": [0.06, 0.26, 0.14, 0.57],
        "Operating_Efficiency": [0.19, 0.63, 0.71, 0.54],
        "Dividend_Payout": [0.0, 0.0, 0.0, 0.0],
        "Yearly_Price_Change": [0.0, 0.0, 0.0, 0.0],
        "Composite_Rank": [0.6, 0.2, 0.26, 0.34],
        "Net_Debt_to_Equity": [0.2, 0.11, 1.79, 0.12],
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
