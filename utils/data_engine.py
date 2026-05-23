# utils/data_engine.py
import pandas as pd

# Mengintegrasikan hasil panen data dari 3 notebook Kaggle lu
MASTER_STOCK_DATA = {
    "ANTM": {
        "ticker_jk": "ANTM.JK",
        "name": "Aneka Tambang Tbk.",
        "pred_return": 0.2618,      # Hasil XGBoost & GRU (Pilar 1)
        "sentiment_score": 0.0101,  # Hasil IndoBERT MaxAbsScaler (Pilar 2)
        "graham_price": 3148.69,    # Hasil perhitungan Harga Wajar (Pilar 3)
        "market_price": 3640.0,     # Dari stock-prices.csv
        "piotroski_score": 8,       # Dari key-statistics.csv
        "piotroski_fuzzy": 1.0,
        "debt_to_equity": 0.30,     # Rasio leverage keuangan
        "volatility": 0.15          # Metrik risiko teknikal
    },
    "BRMS": {
        "ticker_jk": "BRMS.JK",
        "name": "Bumi Resources Minerals Tbk.",
        "pred_return": 0.0240,
        "sentiment_score": 0.0090,
        "graham_price": 138.86,
        "market_price": 735.0,
        "piotroski_score": 8,
        "piotroski_fuzzy": 1.0,
        "debt_to_equity": 0.11,
        "volatility": 0.22
    },
    "MDKA": {
        "ticker_jk": "MDKA.JK",
        "name": "Merdeka Copper Gold Tbk.",
        "pred_return": -0.1500,
        "sentiment_score": -0.0609,
        "graham_price": 0.00,       # Terpotong karena EPS negatif
        "market_price": 3260.0,
        "piotroski_score": 6,
        "piotroski_fuzzy": -1.0,
        "debt_to_equity": 1.79,
        "volatility": 0.28
    },
    "PSAB": {
        "ticker_jk": "PSAB.JK",
        "name": "J Resources Asia Pasifik Tbk.",
        "pred_return": 0.0434,
        "sentiment_score": 0.0335,
        "graham_price": 297.24,
        "market_price": 510.0,
        "piotroski_score": 7,
        "piotroski_fuzzy": 0.0,
        "debt_to_equity": 0.57,
        "volatility": 0.19
    }
}

def dapatkan_data_emiten(ticker):
    clean_ticker = ticker.split(".")[0]
    return MASTER_STOCK_DATA.get(clean_ticker, None)