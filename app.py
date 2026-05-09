import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.fetch_stock import fetch_stock_data, fetch_gold_price
from utils.fetch_news import fetch_news
from utils.sentiment import analyze_sentiment
from utils.feature_engineering import build_feature_row, calculate_technical_features, merge_gold_features
from utils.recommendation import generate_recommendation

# ─── CONFIG ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="GoldStock Insight",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "rf_model.pkl")
FUNDAMENTAL_PATH = os.path.join(BASE_DIR, "data", "fundamental_clean.csv")

TICKER_OPTIONS = ["ANTM", "MDKA", "BRMS"]
GOAL_OPTIONS = ["Jangka Pendek", "Jangka Panjang"]

REC_COLORS = {
    "Jangka Panjang":      "#10B981", # Vibrant Emerald Green
    "Jangka Pendek":       "#3B82F6", # Vibrant Blue
    "Overhyped / Hindari": "#F59E0B", # Vibrant Amber
    "Tidak Disarankan":    "#EF4444", # Vibrant Red
}

# ─── CUSTOM CSS (ULTRA PREMIUM & DYNAMIC THEME) ────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, div, p, span, h1, h2, h3, h4, h5, h6 {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Dynamic Animated Background */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.stApp { 
    background: linear-gradient(-45deg, #f8fafc, #e0f2fe, #fef9c3, #eff6ff);
    background-size: 400% 400%;
    animation: gradientBG 20s ease infinite;
    color: #1E293B; 
}

/* Hero Section (Home) */
.hero-container {
    background: rgba(255, 255, 255, 0.4);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 32px;
    padding: 80px 40px;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 30px 60px -15px rgba(37, 99, 235, 0.15), inset 0 2px 20px rgba(255,255,255,0.8);
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #F59E0B, #FBBF24);
    color: #FFFFFF;
    font-size: 0.85rem;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 10px 28px;
    border-radius: 50px;
    margin-bottom: 24px;
    box-shadow: 0 8px 20px rgba(245, 158, 11, 0.4);
}
.hero-title {
    font-size: 4.5rem;
    font-weight: 800;
    color: #0F172A;
    line-height: 1.1;
    margin-bottom: 24px;
    letter-spacing: -1.5px;
}
.hero-title span { 
    background: linear-gradient(135deg, #2563EB, #06B6D4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.25rem;
    color: #475569;
    max-width: 700px;
    margin: 0 auto 30px;
    line-height: 1.8;
    font-weight: 500;
}

/* Feature Cards */
.feature-card {
    background: rgba(255, 255, 255, 0.65);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-radius: 24px;
    padding: 40px 24px;
    text-align: center;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    height: 100%;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.05);
}
.feature-card:hover {
    transform: translateY(-10px) scale(1.02);
    box-shadow: 0 30px 40px -15px rgba(37, 99, 235, 0.15);
    background: rgba(255, 255, 255, 0.85);
    border-color: #BFDBFE;
}
.feature-icon-wrapper {
    width: 72px;
    height: 72px;
    border-radius: 20px;
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    color: #2563EB;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    font-weight: 800;
    margin: 0 auto 24px;
    box-shadow: inset 0 2px 4px rgba(255,255,255,0.8), 0 8px 16px rgba(37, 99, 235, 0.1);
}

/* Section Containers */
.card {
    background: rgba(255, 255, 255, 0.7);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 24px;
    padding: 32px;
    margin-bottom: 1.5rem;
    box-shadow: 0 15px 35px -10px rgba(0, 0, 0, 0.05);
}
.card-gold {
    background: linear-gradient(135deg, rgba(255, 255, 255, 0.9) 0%, rgba(248, 250, 252, 0.8) 100%);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 1);
    border-radius: 24px;
    padding: 36px;
    margin-bottom: 2rem;
    box-shadow: 0 20px 40px -10px rgba(37, 99, 235, 0.1), inset 0 0 0 1px rgba(255,255,255,0.5);
}
.section-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #0F172A;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.section-title::before {
    content: '';
    display: inline-block;
    width: 28px;
    height: 6px;
    border-radius: 3px;
    background: linear-gradient(90deg, #3B82F6, #06B6D4);
}

/* Recommendation badge */
.rec-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 12px;
    font-size: 1.6rem;
    font-weight: 800;
    padding: 18px 48px;
    border-radius: 20px;
    margin: 20px 0;
    letter-spacing: 1px;
    text-transform: uppercase;
}

/* News table */
.news-row {
    background: rgba(255, 255, 255, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.9);
    border-radius: 16px;
    padding: 18px 24px;
    margin-bottom: 14px;
    transition: all 0.3s ease;
}
.news-row:hover { 
    border-color: #93C5FD;
    background: rgba(255, 255, 255, 0.9);
    transform: translateX(6px);
    box-shadow: 0 10px 20px -5px rgba(37, 99, 235, 0.1);
}
.news-title { color: #0F172A; font-size: 1.05rem; font-weight: 700; line-height: 1.5; }
.news-meta { color: #64748B; font-size: 0.85rem; margin-top: 8px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px;}

/* Fund metric box */
.fund-box {
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.8);
    border-radius: 20px;
    padding: 24px;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.fund-box:hover {
    background: rgba(255, 255, 255, 0.9);
    border-color: #BAE6FD;
    transform: translateY(-5px);
    box-shadow: 0 15px 30px -10px rgba(14, 165, 233, 0.15);
}
.fund-label { font-size: 0.8rem; color: #64748B; text-transform: uppercase; letter-spacing: 1px; font-weight: 800; }
.fund-value { font-size: 1.5rem; font-weight: 800; color: #0F172A; margin-top: 8px; }

/* Divider */
.gold-divider {
    border: none;
    border-top: 2px dashed #CBD5E1;
    margin: 4rem 0;
    opacity: 0.5;
}

/* Explanation box */
.explanation-box {
    background: linear-gradient(135deg, #F0FDF4, #FFFFFF);
    border-left: 6px solid #10B981;
    border-radius: 0 20px 20px 0;
    padding: 28px 32px;
    line-height: 1.9;
    color: #166534;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 10px 25px -5px rgba(22, 101, 52, 0.05);
}

/* FIXED Selectbox to prevent text cutoff */
div[data-baseweb="select"] > div {
    background: rgba(255, 255, 255, 0.8) !important; 
    border: 2px solid #E2E8F0 !important; 
    border-radius: 14px !important; 
    color: #0F172A !important; 
    box-shadow: 0 4px 10px -2px rgba(0, 0, 0, 0.05) !important;
    transition: all 0.3s ease !important;
    min-height: 48px !important; /* ensures height is sufficient */
}
div[data-baseweb="select"] > div:hover {
    border-color: #3B82F6 !important;
    background: #FFFFFF !important;
}

/* Beautiful Premium Button */
.stButton > button {
    background: linear-gradient(135deg, #2563EB, #06B6D4) !important;
    color: #FFFFFF !important;
    font-weight: 800 !important;
    font-size: 1.1rem !important;
    letter-spacing: 0.5px !important;
    padding: 1rem 2.5rem !important;
    border: none !important;
    border-radius: 16px !important;
    width: 100% !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 10px 25px -5px rgba(37, 99, 235, 0.4) !important;
}
.stButton > button:hover { 
    transform: translateY(-4px) scale(1.02) !important; 
    box-shadow: 0 20px 35px -10px rgba(6, 182, 212, 0.5) !important; 
}
.stButton > button p {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)


# ─── HELPERS ───────────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_fundamental():
    if not os.path.exists(FUNDAMENTAL_PATH):
        return pd.DataFrame()
    df = pd.read_csv(FUNDAMENTAL_PATH)
    df["Ticker"] = df["Ticker"].str.upper().str.strip()
    return df

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

def get_fundamental_dict(ticker: str, fund_df: pd.DataFrame) -> dict:
    row = fund_df[fund_df["Ticker"] == ticker]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()

def make_stock_chart(stock_df: pd.DataFrame, ticker: str) -> go.Figure:
    df = stock_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df["Date"], open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"],
        name="Harga", increasing_line_color="#10B981",
        decreasing_line_color="#EF4444",
        increasing_fillcolor="rgba(16, 185, 129, 0.2)",
        decreasing_fillcolor="rgba(239, 68, 68, 0.2)",
    ), row=1, col=1)

    # MA7
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA7"], name="MA7",
        line=dict(color="#F59E0B", width=2, dash="solid"),
        opacity=0.9,
    ), row=1, col=1)

    # MA30
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["MA30"], name="MA30",
        line=dict(color="#3B82F6", width=2, dash="dot"),
        opacity=0.9,
    ), row=1, col=1)

    # Volume
    colors_vol = ["#10B981" if c >= o else "#EF4444"
                  for c, o in zip(df["Close"], df["Open"])]
    fig.add_trace(go.Bar(
        x=df["Date"], y=df["Volume"], name="Volume",
        marker_color=colors_vol, opacity=0.5,
    ), row=2, col=1)

    fig.update_layout(
        title=dict(text=f"{ticker} — Grafik Harga & Volume",
                   font=dict(color="#1E293B", size=16, family="Plus Jakarta Sans", weight="bold")),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569", family="Plus Jakarta Sans"),
        legend=dict(bgcolor="rgba(255,255,255,0.8)", font=dict(color="#1E293B")),
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,0,0,0.05)", zeroline=False)
    return fig

def make_sentiment_gauge(score: float) -> go.Figure:
    color = "#10B981" if score >= 0.1 else "#EF4444" if score <= -0.1 else "#F59E0B"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"color": color, "size": 42, "family": "Plus Jakarta Sans", "weight": "bold"},
                "suffix": "", "valueformat": ".3f"},
        gauge=dict(
            axis=dict(range=[-1, 1], tickcolor="#94A3B8",
                      tickfont=dict(color="#64748B", size=12)),
            bar=dict(color=color, thickness=0.25),
            bgcolor="rgba(0,0,0,0.03)",
            bordercolor="rgba(0,0,0,0)",
            steps=[
                dict(range=[-1, -0.1], color="rgba(239, 68, 68, 0.15)"),
                dict(range=[-0.1, 0.1], color="rgba(245, 158, 11, 0.1)"),
                dict(range=[0.1, 1],   color="rgba(16, 185, 129, 0.15)"),
            ],
            threshold=dict(line=dict(color=color, width=4), value=score),
        ),
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#475569", family="Plus Jakarta Sans"),
        height=240,
        margin=dict(l=20, r=20, t=20, b=10),
    )
    return fig

# ─── PAGE RENDERS ──────────────────────────────────────────────────────────────

def render_home_page():
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">Capstone Project Data Science</div>
        <div class="hero-title">GoldStock <span>Insight</span></div>
        <p class="hero-subtitle">
            Platform pintar untuk membantu Anda mengambil keputusan investasi pada saham sektor emas di Indonesia. Didukung oleh Machine Learning dan Analisis Sentimen Real-time.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">ML</div>
            <h4 style="color: #0F172A; margin-bottom: 12px; font-weight: 700; font-size: 1.1rem;">Machine Learning</h4>
            <p style="color: #64748B; font-size: 0.95rem; line-height: 1.6;">Menggunakan model Random Forest untuk memprediksi tren return saham berdasarkan data historis dan teknikal terkini.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">NLP</div>
            <h4 style="color: #0F172A; margin-bottom: 12px; font-weight: 700; font-size: 1.1rem;">Sentimen Berita</h4>
            <p style="color: #64748B; font-size: 0.95rem; line-height: 1.6;">Menganalisis sentimen dari ratusan berita terbaru untuk mengetahui persepsi pasar secara instan.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon-wrapper">FA</div>
            <h4 style="color: #0F172A; margin-bottom: 12px; font-weight: 700; font-size: 1.1rem;">Fundamental</h4>
            <p style="color: #64748B; font-size: 0.95rem; line-height: 1.6;">Menilai kesehatan perusahaan (PER, PBV, ROE) agar investasi Anda tetap berpijak pada nilai intrinsik.</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1.2, 1])
    with col_btn2:
        if st.button("Mulai Analisis Sekarang", use_container_width=True):
            st.session_state.page = "analysis"
            st.rerun()

def render_analysis_page():
    st.markdown("""
    <div style="margin-bottom: 2rem;">
    """, unsafe_allow_html=True)
    if st.button("Kembali ke Beranda"):
        st.session_state.page = "home"
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)
        
    st.markdown('<div class="card-gold">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Pilih Saham & Tujuan Investasi</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2, 2, 1.5])
    with col1:
        ticker = st.selectbox("Pilih Saham", TICKER_OPTIONS,
                              help="Pilih saham sektor emas yang ingin dianalisis")
    with col2:
        investment_goal = st.selectbox("Tujuan Investasi", GOAL_OPTIONS,
                                       help="Pilih horizon investasi Anda")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        run = st.button("Jalankan Analisis")

    st.markdown('</div>', unsafe_allow_html=True)

    if not run:
        st.markdown("""
        <div style="text-align:center; padding: 5rem 2rem; color: #64748B; background: rgba(255, 255, 255, 0.6); backdrop-filter: blur(8px); border-radius: 20px; border: 1px dashed #CBD5E1; margin-top: 1rem;">
            <div style="font-size: 1.25rem; font-weight: 700; color: #1E293B; margin-bottom: 8px;">
                Sistem Siap Digunakan
            </div>
            <div style="font-size: 1rem; color: #64748B;">
                Tentukan saham dan tujuan Anda, klik tombol analisis, dan biarkan sistem bekerja untuk Anda.
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    # ── Analysis Flow ─────────────────────────────────────────────────────────
    with st.spinner("Mengambil data pasar terbaru..."):
        stock_df = fetch_stock_data(ticker)

    if stock_df.empty:
        st.error(f"Gagal mengambil data harga saham {ticker}. Coba lagi beberapa saat.")
        return

    with st.spinner("Mengambil harga emas global..."):
        gold_df = fetch_gold_price()

    stock_df_raw = calculate_technical_features(stock_df.copy())
    stock_df = merge_gold_features(stock_df_raw.copy(), gold_df)

    with st.spinner("Mengambil & menganalisis berita terbaru..."):
        news_df = fetch_news(ticker)
        sentiment = analyze_sentiment(news_df)

    # Load fundamental
    fund_df = load_fundamental()
    if fund_df.empty:
        st.warning("File fundamental_clean.csv tidak ditemukan. Analisis fundamental dinonaktifkan.")
        fundamental = {}
    else:
        fundamental = get_fundamental_dict(ticker, fund_df)
        if not fundamental:
            st.warning(f"Data fundamental untuk {ticker} tidak ditemukan dalam CSV.")

    # Prediksi return
    model = load_model()
    predicted_return = 0.0

    if model is not None and fundamental:
        try:
            # Gunakan stock_df_raw (sebelum gold merge) agar build_feature_row bisa merge sendiri
            feature_row = build_feature_row(stock_df_raw.copy(), gold_df, sentiment, fundamental)
            if feature_row is not None:
                predicted_return = float(model.predict(feature_row)[0])
        except Exception as e:
            st.warning(f"Model gagal melakukan prediksi: {e}. Menggunakan rule-based fallback.")
    elif model is None:
        st.info("Model belum dilatih. Jalankan `python train_model.py` terlebih dahulu. Menggunakan estimasi fundamental.")
        predicted_return = fundamental.get("Fundamental_Score", 0.5) * 0.08

    # Generate rekomendasi
    result = generate_recommendation(
        predicted_return=predicted_return,
        sentiment_score=sentiment["Sentiment_Score"],
        fundamental_score=float(fundamental.get("Fundamental_Score", 0.5)),
        investment_goal=investment_goal,
        ticker=ticker,
        fundamental_label=str(fundamental.get("Fundamental_Label", "")),
    )

    st.success("Analisis selesai dengan sukses!")

    # ── HASIL REKOMENDASI ─────────────────────────────────────────────────────
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    rec_color = REC_COLORS.get(result["recommendation"], "#94A3B8")

    st.markdown(f"""
    <div class="card-gold" style="border-left-color: {rec_color}; background: #FFFFFF; box-shadow: 0 15px 35px -5px {rec_color}22;">
        <div class="section-title" style="justify-content: center; margin-bottom: 8px;">Kesimpulan Rekomendasi</div>
        <div style="text-align:center; padding: 10px 0;">
            <div style="font-size: 1rem; color: #64748B; font-weight: 600; margin-bottom: 16px; letter-spacing: 0.5px;">
                {ticker} • {investment_goal}
            </div>
            <div class="rec-badge" style="background: {rec_color}10; border: 2px solid {rec_color}; color: {rec_color}; margin: auto; width: fit-content;">
                {result["recommendation"]}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics row
    last_close = float(stock_df["Close"].iloc[-1]) if not stock_df.empty else 0
    prev_close = float(stock_df["Close"].iloc[-2]) if len(stock_df) > 1 else last_close
    price_chg = ((last_close - prev_close) / prev_close * 100) if prev_close else 0

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""<div class="fund-box">
            <div class="fund-label">Harga Terakhir</div>
            <div class="fund-value">Rp {last_close:,.0f}</div>
            <div style="font-size:0.85rem; font-weight:700; margin-top:6px; color:{'#10B981' if price_chg>=0 else '#EF4444'}">
                {'▲' if price_chg>=0 else '▼'} {abs(price_chg):.2f}%
            </div>
        </div>""", unsafe_allow_html=True)
    with m2:
        ret_color = "#10B981" if result["predicted_return_pct"] >= 3 else "#EF4444"
        st.markdown(f"""<div class="fund-box">
            <div class="fund-label">Prediksi AI Return</div>
            <div class="fund-value" style="color:{ret_color}">{result['predicted_return_pct']:.2f}%</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        sent_color = "#10B981" if sentiment["Sentiment_Label"] == "Positif" else \
                     "#EF4444" if sentiment["Sentiment_Label"] == "Negatif" else "#F59E0B"
        st.markdown(f"""<div class="fund-box">
            <div class="fund-label">Sentimen Berita</div>
            <div class="fund-value" style="color:{sent_color}">{sentiment['Sentiment_Label']}</div>
            <div style="font-size:0.8rem; font-weight:600; color:#94A3B8; margin-top:6px;">Dari {sentiment['News_Count']} sumber</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        risk_color = REC_COLORS.get(result["recommendation"], "#94A3B8")
        st.markdown(f"""<div class="fund-box">
            <div class="fund-label">Level Risiko</div>
            <div class="fund-value" style="color:{risk_color}">{result['risk_level']}</div>
        </div>""", unsafe_allow_html=True)

    # ── GRAFIK HARGA ──────────────────────────────────────────────────────────
    st.markdown('<div class="card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Grafik Harga Saham</div>', unsafe_allow_html=True)
    st.plotly_chart(make_stock_chart(stock_df, ticker), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── SENTIMEN & BERITA ─────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sentimen & Berita Terbaru</div>', unsafe_allow_html=True)

    sc1, sc2 = st.columns([1, 2])
    with sc1:
        st.plotly_chart(make_sentiment_gauge(sentiment["Sentiment_Score"]), use_container_width=True)
        sent_color = "#10B981" if sentiment["Sentiment_Label"] == "Positif" else \
                     "#EF4444" if sentiment["Sentiment_Label"] == "Negatif" else "#F59E0B"
        st.markdown(f"""<div style="text-align:center; margin-top:-15px;">
            <span style="font-size:1.4rem; font-weight:800; color:{sent_color}">
                {sentiment['Sentiment_Label']}
            </span>
            <div style="font-size:0.9rem; font-weight:600; color:#64748B; margin-top:8px;">
                Skor Sentimen: {sentiment['Sentiment_Score']:.3f}
            </div>
        </div>""", unsafe_allow_html=True)

    with sc2:
        if news_df.empty:
            st.info("Tidak ada berita yang berhasil diambil saat ini.")
        else:
            for _, row in news_df.head(6).iterrows():
                date_str = row["Date"].strftime("%d %b %Y") if pd.notna(row["Date"]) else "—"
                st.markdown(f"""
                <div class="news-row">
                    <a href="{row['Link']}" target="_blank" style="text-decoration:none;">
                        <div class="news-title">{row['Title']}</div>
                        <div class="news-meta">{date_str} • {row['Source']}</div>
                    </a>
                </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── FUNDAMENTAL ───────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Analisis Fundamental</div>', unsafe_allow_html=True)

    if not fundamental:
        st.info("Data fundamental belum tersedia untuk saham ini.")
    else:
        fund_score = float(fundamental.get("Fundamental_Score", 0))
        fund_label = str(fundamental.get("Fundamental_Label", "—"))
        fund_label_color = "#10B981" if fund_score >= 0.7 else "#F59E0B" if fund_score >= 0.5 else "#EF4444"

        st.markdown(f"""
        <div style="text-align:center; margin-bottom:30px; padding: 24px; background: #F8FAFC; border-radius: 16px; border: 1px solid #E2E8F0;">
            <span style="font-size:0.95rem; font-weight:700; color:#64748B; text-transform:uppercase; letter-spacing:1.5px;">Skor Kesehatan Fundamental</span><br>
            <div style="margin-top: 12px; display: flex; align-items: center; justify-content: center; gap: 16px;">
                <span style="font-size:3.5rem; font-weight:800; color:{fund_label_color}; line-height:1;">
                    {fund_score:.2f}
                </span>
                <span style="font-size:1.2rem; font-weight:700; color:{fund_label_color}; padding: 6px 16px; background: {fund_label_color}15; border-radius: 30px;">
                    {fund_label}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        f_cols = st.columns(4)
        metrics = [
            ("PER",          fundamental.get("PER", "—"),           "Price to Earnings"),
            ("PBV",          fundamental.get("PBV", "—"),           "Price to Book Value"),
            ("EPS",          fundamental.get("EPS", "—"),           "Earnings per Share"),
            ("ROE",          fundamental.get("ROE", "—"),           "Return on Equity"),
            ("DER",          fundamental.get("DER", "—"),           "Debt to Equity"),
            ("Current Ratio",fundamental.get("Current_Ratio", "—"), "Likuiditas"),
            ("Market Cap",   fundamental.get("Market_Cap", "—"),    "Kapitalisasi Pasar"),
        ]
        for i, (label, value, tooltip) in enumerate(metrics):
            with f_cols[i % 4]:
                try:
                    display_val = f"{float(value):,.2f}"
                except Exception:
                    display_val = str(value)
                st.markdown(f"""<div class="fund-box" title="{tooltip}" style="padding: 16px; margin-bottom: 16px;">
                    <div class="fund-label" style="font-size:0.75rem;">{label}</div>
                    <div class="fund-value" style="font-size:1.25rem;">{display_val}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # ── PENJELASAN ────────────────────────────────────────────────────────────
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Penjelasan Keputusan AI</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="explanation-box">{result["explanation"]}</div>',
                unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)


# ─── MAIN APP ROUTING ──────────────────────────────────────────────────────────

def main():
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.session_state.page == "home":
        render_home_page()
    elif st.session_state.page == "analysis":
        render_analysis_page()

    # Footer always shown
    st.markdown('<hr class="gold-divider">', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding: 20px; color: #64748B; font-size: 0.9rem; line-height: 1.8;">
        <strong style="color: #475569;">Disclaimer</strong><br>
        Aplikasi ini hanya digunakan sebagai alat bantu analisis dan edukasi.<br>
        Hasil rekomendasi bukan merupakan ajakan untuk membeli atau menjual saham.<br>
        Keputusan investasi sepenuhnya menjadi tanggung jawab investor.<br><br>
        <span style="color: #94A3B8; font-weight: 600; letter-spacing: 0.5px;">GOLDSTOCK INSIGHT • CAPSTONE PROJECT 2024</span>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
