import os
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from config import TICKER_MAP, SCALER_FEATURES, GRU_LOOKBACK

class DummyST:
    def cache_resource(self, *args, **kwargs):
        return lru_cache(maxsize=32)
    def cache_data(self, *args, **kwargs):
        return lru_cache(maxsize=32)

st = DummyST()

@st.cache_data(ttl=60)
def ambil_data_asli_kaggle():
    data_dinamis = {}
    try:
        df_fund = pd.read_csv("data/fundamental_evaluasi_final.csv")
    except Exception:
        df_fund = pd.DataFrame()

    daftar_saham = ["ANTM", "BRMS", "MDKA", "PSAB"]
    
    for ticker in daftar_saham:
        data_dinamis[ticker] = {
            "pred_return": 0.02, 
            "sentiment_score": 0.0,
            "graham_price": 0.0,
            "piotroski_score": 5,
            "piotroski_fuzzy": 0.0
        }
        if not df_fund.empty and ticker in df_fund['Ticker'].values:
            row_fund = df_fund[df_fund['Ticker'] == ticker].iloc[0]
            data_dinamis[ticker]["graham_price"] = row_fund.get("Harga_Wajar_Graham", 0.0)
            data_dinamis[ticker]["piotroski_score"] = row_fund.get("Piotroski F-Score", 5)
            data_dinamis[ticker]["piotroski_fuzzy"] = row_fund.get("Skor_Piotroski_Fuzzy", 0.0)
            
        file_nlp = f"data/sentimen_{ticker}.csv" 
        if os.path.exists(file_nlp):
            df_nlp = pd.read_csv(file_nlp)
            if not df_nlp.empty:
                data_dinamis[ticker]["sentiment_score"] = float(df_nlp.iloc[0].get("Skor_Sentimen_Final", 0.0))

    return data_dinamis

KAGGLE_PILAR_DATA = ambil_data_asli_kaggle()

@st.cache_resource(ttl=3600)
def load_ml_models(ticker):
    ticker_jk = TICKER_MAP.get(ticker, ticker + ".JK")
    gru_path = f"models/gru_{ticker_jk}.h5"
    xgb_path = f"models/xgb_{ticker_jk}.json"
    scaler_path = f"models/scaler_{ticker_jk}.pkl"

    result = {}

    if os.path.exists(scaler_path):
        try:
            result["scaler"] = joblib.load(scaler_path)
        except Exception:
            pass

    if os.path.exists(gru_path):
        try:
            from tensorflow import keras
            class CustomAttention(keras.layers.Attention):
                def __init__(self, *args, **kwargs):
                    if 'score_mode' in kwargs and callable(kwargs['score_mode']):
                        kwargs['score_mode'] = 'dot'
                    super().__init__(*args, **kwargs)

            result["gru"] = keras.models.load_model(
                gru_path, 
                custom_objects={'Attention': CustomAttention},
                compile=False
            )
        except Exception:
            pass

    if os.path.exists(xgb_path):
        try:
            import xgboost as xgb_lib
            xgb_model = xgb_lib.XGBRegressor()
            xgb_model.load_model(xgb_path)
            result["xgb"] = xgb_model
        except Exception:
            pass

    return result if result else None

def build_gru_window(stock_df, gold_df, scaler, ticker):
    try:
        merged = pd.merge(
            stock_df,
            gold_df[["Date", "Gold_Close", "Gold_Return"]],
            on="Date",
            how="left"
        ).ffill().dropna(subset=["Close"]).reset_index(drop=True)

        for col in SCALER_FEATURES:
            if col not in merged.columns:
                merged[col] = 0.0

        window_df = merged[SCALER_FEATURES].tail(GRU_LOOKBACK).copy()
        window_df = window_df.ffill().fillna(0.0)

        if len(window_df) < GRU_LOOKBACK:
            pad_rows = GRU_LOOKBACK - len(window_df)
            pad_df = pd.concat([window_df.iloc[[0]] * pad_rows, window_df], ignore_index=True)
            window_df = pad_df

        X_raw = window_df.values.astype(np.float32)
        X_scaled = scaler.transform(X_raw)
        X_gru = X_scaled.reshape(1, GRU_LOOKBACK, len(SCALER_FEATURES))
        
        return X_gru
    except Exception as e:
        raise e

def fallback_predict_return(row):
    technical_signal = 0
    if row["Close"] > row["MA7"]: technical_signal += 0.008
    else: technical_signal -= 0.004

    if row["MA7"] > row["MA30"]: technical_signal += 0.012
    else: technical_signal -= 0.006

    sentiment_signal = row["Sentiment_Score"] * 0.012
    fundamental_signal = (row["Composite_Rank"] - 0.5) * 0.025
    gold_signal = row["Gold_Return"] * 0.6 if not pd.isna(row["Gold_Return"]) else 0
    volatility_penalty = min(row["Volatility"], 0.08) * 0.15 if not pd.isna(row["Volatility"]) else 0

    predicted = technical_signal + sentiment_signal + fundamental_signal + gold_signal - volatility_penalty
    return float(np.clip(predicted, -0.08, 0.12))

def predict_return(ticker, latest_row, stock_df=None, gold_df=None):
    models = load_ml_models(ticker)
    gru_model = models.get("gru") if models else None
    xgb_model = models.get("xgb") if models else None
    scaler    = models.get("scaler") if models else None

    if gru_model is not None and scaler is not None and stock_df is not None and gold_df is not None:
        X_gru = build_gru_window(stock_df, gold_df, scaler, ticker)
        if X_gru is not None:
            from tensorflow import keras
            bottleneck_model = keras.Model(
                inputs=gru_model.input,
                outputs=gru_model.layers[-2].output
            )
            gru_features = bottleneck_model.predict(X_gru, verbose=0)
            if xgb_model is not None:
                if gru_features.shape[1] != xgb_model.n_features_in_:
                    raise ValueError(f"XGBoost expected {xgb_model.n_features_in_} features, but got {gru_features.shape[1]}")
                pred_xgb = float(xgb_model.predict(gru_features)[0])
                return pred_xgb, "Machine Learning (GRU+XGBoost)"
            else:
                raise ValueError("XGBoost model is missing!")

    val = fallback_predict_return(latest_row)
    return val, "Rule-Based Evaluation"
