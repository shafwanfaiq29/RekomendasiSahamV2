# utils/decision_support.py
import pandas as pd
from utils.data_engine import MASTER_STOCK_DATA
from utils.fuzzy_engine import hitung_rekomendasi_fuzzy

def deteksi_overhyped(data_emiten):
    """
    Fakta Finansial: Jika harga pasar jauh di atas harga wajar Graham,
    namun sentimen berita sangat euforia, emiten dikategorikan Overhyped (FOMO).
    """
    if data_emiten["graham_price"] == 0:
        return "HIGH RISK (EPS MINUS)"
    
    rasio_valuasi = data_emiten["market_price"] / data_emiten["graham_price"]
    if rasio_valuasi > 1.3 and data_emiten["sentiment_score"] > 0:
        return "OVERHYPED (FOMO BUBBLE)"
    elif rasio_valuasi < 0.9:
        return "UNDERVALUED (DISKON)"
    return "FAIR VALUE (WAJAR)"

def hitung_risk_level(data_emiten):
    """
    Kombinasi risiko berdasarkan volatilitas harga dan debt-to-equity ratio.
    """
    if data_emiten["debt_to_equity"] > 1.0 or data_emiten["volatility"] > 0.25:
        return "HIGH RISK"
    elif data_emiten["debt_to_equity"] > 0.5:
        return "MODERATE RISK"
    return "LOW RISK"

def bangun_dashboard_metrics():
    """
    Mengompilasi seluruh metrik untuk Watchlist dan Top Recommendation.
    """
    watchlist_items = []
    for ticker, data in MASTER_STOCK_DATA.items():
        skor_fuzzy = hitung_rekomendasi_fuzzy(data["pred_return"], data["sentiment_score"], data["piotroski_fuzzy"])
        
        if skor_fuzzy >= 65:
            rec = "JANGKA PANJANG"
        elif skor_fuzzy >= 40:
            rec = "JANGKA PENDEK"
        else:
            rec = "HINDARI"
            
        watchlist_items.append({
            "Ticker": ticker,
            "Nama": data["name"],
            "Harga_Pasar": data["market_price"],
            "Fuzzy_Score": skor_fuzzy,
            "Rekomendasi": rec,
            "Risk": hitung_risk_level(data),
            "Hype": deteksi_overhyped(data)
        })
    
    # Urutkan berdasarkan skor fuzzy tertinggi untuk fitur Top Recommendation
    df_metrics = pd.DataFrame(watchlist_items).sort_values(by="Fuzzy_Score", ascending=False).reset_index(drop=True)
    return df_metrics