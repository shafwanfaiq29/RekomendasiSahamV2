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
    
    # 1. Buka harta karun Pilar 3 (Fundamental Piotroski & Graham)
    try:
        df_fund = pd.read_csv("data/fundamental_evaluasi_final.csv")
    except Exception:
        df_fund = pd.DataFrame()

    daftar_saham = ["ANTM", "BRMS", "MDKA", "PSAB"]
    
    for ticker in daftar_saham:
        # Template default jika file gagal terbaca
        data_dinamis[ticker] = {
            "pred_return": 0.02, 
            "sentiment_score": 0.0,
            "graham_price": 0.0,
            "piotroski_score": 5,
            "piotroski_fuzzy": 0.0
        }
        
        # --- INJEKSI DATA FUNDAMENTAL ---
        if not df_fund.empty and ticker in df_fund['Ticker'].values:
            row_fund = df_fund[df_fund['Ticker'] == ticker].iloc[0]
            data_dinamis[ticker]["graham_price"] = row_fund.get("Harga_Wajar_Graham", 0.0)
            data_dinamis[ticker]["piotroski_score"] = row_fund.get("Piotroski F-Score", 5)
            data_dinamis[ticker]["piotroski_fuzzy"] = row_fund.get("Skor_Piotroski_Fuzzy", 0.0)
            
        # --- INJEKSI DATA SENTIMEN INDOBERT ---
        # Sesuaiin format nama file dengan yang lu export kemarin (misal: sentimen_ANTM.JK.csv)
        file_nlp = f"data/sentimen_{ticker}.csv" 
        if os.path.exists(file_nlp):
            df_nlp = pd.read_csv(file_nlp)
            if not df_nlp.empty:
                # Ambil skor sentimen hari terbaru (baris ke-0)
                data_dinamis[ticker]["sentiment_score"] = float(df_nlp.iloc[0].get("Skor_Sentimen_Final", 0.0))

        # --- INJEKSI DATA PREDIKSI TIME-SERIES ---
        # XGBoost lu udah ke-handle otomatis sama fungsi load_model("models/rf_model.pkl") 
        # yang ada di app.py lu. Jadi biarin aja fungsi predict_return bawaan lu bekerja.
        
    return data_dinamis

KAGGLE_PILAR_DATA = ambil_data_asli_kaggle()

@st.cache_resource(ttl=3600)
def load_ml_models(ticker):
    """
    Load GRU (Keras .h5), XGBoost (native .json), dan Scaler (.pkl)
    untuk ticker yang diberikan.

    File convention:
        models/gru_{TICKER}.JK.h5
        models/xgb_{TICKER}.JK.json       <- XGB takes GRU bottleneck embedding
        models/scaler_{TICKER}.JK.pkl     <- fitted on 11 SCALER_FEATURES
    """
    ticker_jk = TICKER_MAP.get(ticker, ticker + ".JK")
    gru_path = f"models/gru_{ticker_jk}.h5"
    xgb_path = f"models/xgb_{ticker_jk}.json"
    scaler_path = f"models/scaler_{ticker_jk}.pkl"

    print(f"[DEBUG] Selected ticker  : {ticker}")
    print(f"[DEBUG] GRU model path   : {gru_path}")
    print(f"[DEBUG] XGB model path   : {xgb_path}")
    print(f"[DEBUG] Scaler path      : {scaler_path}")

    result = {}

    # --- Load Scaler (11 features) ---
    if os.path.exists(scaler_path):
        try:
            result["scaler"] = joblib.load(scaler_path)
            print(f"[DEBUG] Scaler loaded OK : {scaler_path} (n_features={result['scaler'].n_features_in_})")
        except Exception as e:
            print(f"[ERROR] Scaler load failed: {e}")
    else:
        print(f"[WARN]  Scaler not found : {scaler_path}")

    # --- Load GRU (Keras .h5) ---
    if os.path.exists(gru_path):
        try:
            from tensorflow import keras
            
            # Patch Attention layer to handle callable score_mode saved in the model
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
            print(f"[DEBUG] GRU loaded OK    : {gru_path} input={result['gru'].input_shape}")
        except Exception as e:
            print(f"[ERROR] GRU load failed  : {e}")
    else:
        print(f"[WARN]  GRU not found    : {gru_path}")

    # --- Load XGBoost (native JSON, takes GRU bottleneck embedding) ---
    if os.path.exists(xgb_path):
        try:
            import xgboost as xgb_lib
            xgb_model = xgb_lib.XGBRegressor()
            xgb_model.load_model(xgb_path)
            result["xgb"] = xgb_model
            print(f"[DEBUG] XGB loaded OK    : {xgb_path}")
        except Exception as e:
            print(f"[ERROR] XGB load failed  : {e}")
    else:
        print(f"[WARN]  XGB not found    : {xgb_path}")

    print("===== MODEL LOAD CHECK =====")
    print("Ticker:", ticker)
    print("GRU Exists:", os.path.exists(gru_path))
    print("XGB Exists:", os.path.exists(xgb_path))
    print("Scaler Exists:", os.path.exists(scaler_path))
    print("GRU Loaded:", result.get("gru") is not None)
    print("XGB Loaded:", result.get("xgb") is not None)
    print("Scaler Loaded:", result.get("scaler") is not None)

    return result if result else None

def build_gru_window(stock_df, gold_df, scaler, ticker):
    """
    Bangun window 60-hari untuk input GRU.
    Mengembalikan array shape (1, 60, 11) atau None jika data kurang.
    """
    try:
        # Merge stock + gold
        merged = pd.merge(
            stock_df,
            gold_df[["Date", "Gold_Close", "Gold_Return"]],
            on="Date",
            how="left"
        ).ffill().dropna(subset=["Close"]).reset_index(drop=True)

        # Pastikan semua SCALER_FEATURES tersedia
        for col in SCALER_FEATURES:
            if col not in merged.columns:
                merged[col] = 0.0

        window_df = merged[SCALER_FEATURES].tail(GRU_LOOKBACK).copy()
        window_df = window_df.ffill().fillna(0.0)

        if len(window_df) < GRU_LOOKBACK:
            print(f"[WARN]  Data hanya {len(window_df)} baris, GRU butuh {GRU_LOOKBACK}")
            # Pad with first row if not enough data
            pad_rows = GRU_LOOKBACK - len(window_df)
            pad_df = pd.concat([window_df.iloc[[0]] * pad_rows, window_df], ignore_index=True)
            window_df = pad_df

        X_raw = window_df.values.astype(np.float32)          # (60, 11)
        
        print("Scaler Columns Used:", SCALER_FEATURES)
        print("Incoming Columns:", list(window_df.columns))
        
        X_scaled = scaler.transform(X_raw)                    # (60, 11)
        X_gru = X_scaled.reshape(1, GRU_LOOKBACK, len(SCALER_FEATURES))  # (1, 60, 11)
        
        print("===== PREDICTION INPUT CHECK =====")
        print("Ticker:", ticker)
        print("Scaler Feature Shape:", X_scaled.shape)
        print("GRU Window Shape:", X_gru.shape)
        print("Feature Columns:", SCALER_FEATURES)
        
        return X_gru
    except Exception as e:
        print("ML prediction error:", e)
        raise e

def predict_return(ticker, latest_row, stock_df=None, gold_df=None):
    """
    Prediksi return saham dengan prioritas:
      1. GRU (60-step window)  — jika berhasil load & predict
      2. XGBoost (GRU bottleneck) — jika GRU predict berhasil ambil embedding
      3. Rule-based             — hanya jika semua model gagal
    """
    models = load_ml_models(ticker)

    # ====================================================
    # PRIORITY 1+2 — GRU → XGBoost stacking
    # ====================================================
    gru_model = models.get("gru") if models else None
    xgb_model = models.get("xgb") if models else None
    scaler    = models.get("scaler") if models else None

    if gru_model is not None and scaler is not None and stock_df is not None and gold_df is not None:
        X_gru = build_gru_window(stock_df, gold_df, scaler, ticker)
        if X_gru is not None:
            # Removed try-except to expose original errors
            # Extract GRU bottleneck: use intermediate layer output
            from tensorflow import keras
            # Get the penultimate layer (bottleneck before dense output)
            bottleneck_model = keras.Model(
                inputs=gru_model.input,
                outputs=gru_model.layers[-2].output
            )
            gru_features = bottleneck_model.predict(X_gru, verbose=0)  # (1, N)
            
            print("GRU Embedding Shape:", gru_features.shape)
            print("GRU Embedding:", gru_features[:5])
            
            if xgb_model is not None:
                print("XGB Input Shape:", gru_features.shape)
                print("XGB Expected Features:", xgb_model.n_features_in_)
                
                if gru_features.shape[1] != xgb_model.n_features_in_:
                    raise ValueError(f"XGBoost expected {xgb_model.n_features_in_} features, but got {gru_features.shape[1]}")
                
                pred_xgb = float(xgb_model.predict(gru_features)[0])
                
                print("===== XGBOOST OUTPUT =====")
                print("Predicted Return:", pred_xgb)
                print("Prediction Type:", type(pred_xgb))
                
                prediction_source = "Machine Learning (GRU+XGBoost)"
                print("===== FINAL SOURCE =====")
                print("Prediction Source:", prediction_source)
                return pred_xgb, prediction_source
            else:
                raise ValueError("XGBoost model is missing!")

    # ====================================================
    # PRIORITY 3 — Rule-based fallback
    # ====================================================
    print("[WARN] ML models failed to load or missing dependencies. Using Rule-Based fallback.")
    val = fallback_predict_return(latest_row)
    print(f"[DEBUG] Prediction source  : Rule-Based Evaluation")
    print(f"[DEBUG] Predicted return   : {val:.6f}")
    return val, "Rule-Based Evaluation"

def fallback_predict_return(row):
    """
    Fallback kalau model belum tersedia.
    Ini bukan model ML, tapi rule-based signal agar app tetap bisa demo.
    """
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
