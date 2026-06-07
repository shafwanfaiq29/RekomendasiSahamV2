import numpy as np
import pandas as pd
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from config import TICKER_MAP, COMPANY_NAMES
from utils.data_fetcher import fetch_stock_data, fetch_gold_data, load_fundamental_data, fetch_news
from utils.sentiment_analyzer import apply_sentiment
from utils.model_predictor import predict_return, KAGGLE_PILAR_DATA
from functools import lru_cache

class DummyST:
    def cache_data(self, *args, **kwargs):
        return lru_cache(maxsize=32)
        
st = DummyST()

def eksekusi_fuzzy_mamdani(prediksi_return, skor_sentimen, skor_fundamental):
    universe_1 = np.arange(-0.11, 0.12, 0.001)
    universe_2 = np.arange(-1.01, 1.02, 0.01)
    
    in_return = ctrl.Antecedent(universe_1, 'return_ai')
    in_sentimen = ctrl.Antecedent(universe_2, 'sentimen')
    in_fund = ctrl.Antecedent(universe_2, 'fundamental')
    out_keputusan = ctrl.Consequent(np.arange(0, 101, 1), 'rekomendasi')
    
    in_return['bearish'] = fuzz.trimf(universe_1, [-10.01, -10.01, 0])
    in_return['stagnan'] = fuzz.trimf(universe_1, [-5, 0, 5])
    in_return['bullish'] = fuzz.trimf(universe_1, [0, 10.01, 10.01])
    
    in_sentimen['negatif'] = fuzz.trimf(universe_2, [-1.01, -1.01, 0])
    in_sentimen['netral'] = fuzz.trimf(universe_2, [-0.5, 0, 0.5])
    in_sentimen['positif'] = fuzz.trimf(universe_2, [0, 1.01, 1.01])
    
    in_fund['sakit'] = fuzz.trimf(universe_2, [-1.01, -1.01, 0])
    in_fund['biasa'] = fuzz.trimf(universe_2, [-0.5, 0, 0.5])
    in_fund['sehat'] = fuzz.trimf(universe_2, [0, 1.01, 1.01])
    
    out_keputusan['hindari'] = fuzz.trimf(out_keputusan.universe, [0, 0, 45])
    out_keputusan['jangka_pendek'] = fuzz.trimf(out_keputusan.universe, [35, 50, 65])
    out_keputusan['jangka_panjang'] = fuzz.trimf(out_keputusan.universe, [55, 100, 100])
    
    # Complete rules matrix
    r1 = ctrl.Rule(in_fund['sakit'], out_keputusan['hindari'])
    r2 = ctrl.Rule(in_sentimen['negatif'], out_keputusan['hindari'])
    r3 = ctrl.Rule(in_return['bearish'] & in_fund['biasa'], out_keputusan['hindari'])
    
    r4 = ctrl.Rule(in_return['bearish'] & in_fund['sehat'], out_keputusan['jangka_pendek'])
    r5 = ctrl.Rule(in_return['stagnan'] & in_fund['biasa'], out_keputusan['jangka_pendek'])
    r6 = ctrl.Rule(in_return['stagnan'] & in_fund['sehat'], out_keputusan['jangka_pendek'])
    r7 = ctrl.Rule(in_return['bullish'] & in_fund['biasa'], out_keputusan['jangka_pendek'])
    
    r8 = ctrl.Rule(in_return['bullish'] & in_fund['sehat'] & in_sentimen['positif'], out_keputusan['jangka_panjang'])
    r9 = ctrl.Rule(in_return['bullish'] & in_fund['sehat'] & in_sentimen['netral'], out_keputusan['jangka_panjang'])
    
    mesin = ctrl.ControlSystem([r1, r2, r3, r4, r5, r6, r7, r8, r9])
    simulasi = ctrl.ControlSystemSimulation(mesin)
    
    simulasi.input['return_ai'] = np.clip(prediksi_return * 100, -10, 10)
    simulasi.input['sentimen'] = np.clip(skor_sentimen, -1, 1)
    simulasi.input['fundamental'] = np.clip(skor_fundamental, -1, 1)
    
    try:
        simulasi.compute()
        return simulasi.output['rekomendasi']
    except:
        return 50.0

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

def generate_recommendation(ticker, predicted_return, sentiment_score, fundamental_score, investment_goal):
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
    risk_score = 50
    if fundamental_score >= 0.5: risk_score -= 20
    elif fundamental_score < 0: risk_score += 30
        
    if volatility and not pd.isna(volatility):
        if volatility > 0.05: risk_score += 25
        elif volatility < 0.02: risk_score -= 10
            
    if sentiment_score < -0.1: risk_score += 15
    elif sentiment_score > 0.1: risk_score -= 10
        
    if investment_goal == "Jangka Pendek": risk_score += 10
        
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
            
            data = KAGGLE_PILAR_DATA.get(ticker_name)
            if data:
                fund_dict["Composite_Rank"] = data["piotroski_fuzzy"]
                # avg_score = data["sentiment_score"]
                # sentiment_label = "Positive" if avg_score > 0 else "Negative" if avg_score < 0 else "Neutral"
                # fund_dict["Composite_Rank"] = data["piotroski_fuzzy"]
            
            fundamental_score = fund_dict.get("Composite_Rank", 0)
            fund_label = "Kuat / Good" if fundamental_score >= 0 else "Lemah / Weak"
            
            predicted_return, model_source = predict_return(ticker_name, latest_row, stock_df=stock_df, gold_df=gold_df)
            
            rec, risk, overall_score = generate_recommendation(
                ticker_name, predicted_return, avg_score, fundamental_score, "Jangka Pendek"
            )
            
            risk_level, risk_score, _ = calculate_risk_level(latest_row.get("Volatility", 0), predicted_return, avg_score, fundamental_score, "Jangka Pendek")
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
        except Exception:
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
