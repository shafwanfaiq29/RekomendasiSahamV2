import os
import requests
import json
import joblib
import numpy as np
import pandas as pd
from functools import lru_cache
from config import TICKER_MAP, SCALER_FEATURES, GRU_LOOKBACK

# ==============================================================================
# BASE_DIR: path absolut ke root proyek agar berjalan di Linux/Docker maupun Windows
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DummyST:
    def cache_resource(self, *args, **kwargs):
        return lru_cache(maxsize=32)
    def cache_data(self, *args, **kwargs):
        return lru_cache(maxsize=32)

st = DummyST()

# ==============================================================================
# KONFIGURASI API — diambil dari environment variable, TIDAK hardcoded
# ==============================================================================
MODEL_API_URL = os.getenv("MODEL_API_URL", "")
HF_API_KEY    = os.getenv("HF_API_KEY", "")


@st.cache_data(ttl=60)
def ambil_data_asli_kaggle():
    """
    Baca data fundamental dari Excel (semua 8 ticker) dan data sentimen NLP
    jika file tersedia.

    Kolom yang diambil dari get_fundamental_row():
      - Harga_Wajar_Graham      → graham_price
      - Piotroski_F_Score       → piotroski_score  (nama kolom sudah distandarisasi)
      - Skor_Piotroski_Fuzzy    → piotroski_fuzzy  (skala -1/+1, untuk fuzzy Mamdani)

    Tidak ada override Composite_Rank di sini — setiap fungsi menerima variabel
    yang tepat dari get_fundamental_row() secara langsung di app.py.
    """
    from utils.data_fetcher import get_fundamental_row

    data_dinamis = {}
    daftar_saham = ["ANTM", "BRMS", "MDKA", "PSAB", "UNTR", "HRTA", "ARCI", "AMMN"]

    for ticker in daftar_saham:
        # Default
        data_dinamis[ticker] = {
            "pred_return":      0.02,
            "sentiment_score":  0.0,
            "graham_price":     0.0,
            "piotroski_score":  5,
            "piotroski_fuzzy":  0.0,
        }

        # Ambil fundamental via single source of truth
        fund = get_fundamental_row(ticker)
        if fund and fund.get('Ticker') == ticker:
            data_dinamis[ticker]["graham_price"]    = fund.get("Harga_Wajar_Graham", 0.0)
            data_dinamis[ticker]["piotroski_score"] = fund.get("Piotroski_F_Score", 5)
            # Skor_Piotroski_Fuzzy: -1/+1, ini yang benar untuk fuzzy Mamdani
            data_dinamis[ticker]["piotroski_fuzzy"] = fund.get("Skor_Piotroski_Fuzzy", 0.0)

        # Data sentimen NLP jika tersedia
        file_nlp = os.path.join(BASE_DIR, "data", f"sentimen_{ticker}.csv")
        if os.path.exists(file_nlp):
            df_nlp = pd.read_csv(file_nlp)
            if not df_nlp.empty:
                data_dinamis[ticker]["sentiment_score"] = float(
                    df_nlp.iloc[0].get("Skor_Sentimen_Final", 0.0)
                )

    return data_dinamis


KAGGLE_PILAR_DATA = ambil_data_asli_kaggle()


@st.cache_resource(ttl=3600)
def load_ml_models(ticker):
    ticker_jk   = TICKER_MAP.get(ticker, ticker + ".JK")
    gru_path    = os.path.join(BASE_DIR, "models", f"gru_{ticker_jk}.h5")
    xgb_path    = os.path.join(BASE_DIR, "models", f"xgb_{ticker_jk}.json")
    scaler_path = os.path.join(BASE_DIR, "models", f"scaler_{ticker_jk}.pkl")

    print(f"[DEBUG] Ticker: {ticker} | GRU: {gru_path} | XGB: {xgb_path}")

    result = {}

    if os.path.exists(scaler_path):
        try:
            result["scaler"] = joblib.load(scaler_path)
            print(f"[DEBUG] Scaler loaded (n_features={result['scaler'].n_features_in_})")
        except Exception as e:
            print(f"[ERROR] Scaler: {e}")

    if os.path.exists(gru_path):
        try:
            from tensorflow import keras
            class CustomAttention(keras.layers.Attention):
                def __init__(self, *args, **kwargs):
                    if 'score_mode' in kwargs and callable(kwargs['score_mode']):
                        kwargs['score_mode'] = 'dot'
                    super().__init__(*args, **kwargs)
            result["gru"] = keras.models.load_model(
                gru_path, custom_objects={'Attention': CustomAttention}, compile=False
            )
            print(f"[DEBUG] GRU loaded input={result['gru'].input_shape}")
        except Exception as e:
            print(f"[ERROR] GRU: {e}")

    if os.path.exists(xgb_path):
        try:
            import xgboost as xgb_lib
            xgb_model = xgb_lib.XGBRegressor()
            xgb_model.load_model(xgb_path)
            result["xgb"] = xgb_model
            print(f"[DEBUG] XGB loaded")
        except Exception as e:
            print(f"[ERROR] XGB: {e}")

    return result if result else None


def build_gru_window(stock_df, gold_df, scaler, ticker):
    try:
        merged = pd.merge(
            stock_df,
            gold_df[["Date", "Gold_Close", "Gold_Return"]],
            on="Date", how="left"
        ).ffill().dropna(subset=["Close"]).reset_index(drop=True)

        for col in SCALER_FEATURES:
            if col not in merged.columns:
                merged[col] = 0.0

        window_df = merged[SCALER_FEATURES].tail(GRU_LOOKBACK).copy().ffill().fillna(0.0)

        if len(window_df) < GRU_LOOKBACK:
            pad_rows = GRU_LOOKBACK - len(window_df)
            window_df = pd.concat(
                [window_df.iloc[[0]] * pad_rows, window_df], ignore_index=True
            )

        X_scaled = scaler.transform(window_df.values.astype(np.float32))
        return X_scaled.reshape(1, GRU_LOOKBACK, len(SCALER_FEATURES))
    except Exception as e:
        print(f"[ERROR] build_gru_window: {e}")
        raise


def predict_via_api(ticker, latest_row, stock_df, gold_df):
    """
    Kirim data ke Hugging Face Inference API (GPU endpoint) via HTTP POST.
    Gunakan MODEL_API_URL dan HF_API_KEY dari environment variable.

    Payload dikirim sebagai JSON; response diharapkan berisi {"predicted_return": float}.
    Jika API tidak tersedia atau gagal, kembalikan None agar fallback ke rule-based.
    """
    if not MODEL_API_URL or not HF_API_KEY:
        return None

    try:
        # Serialize latest_row (Series/dict) ke dict primitif
        if hasattr(latest_row, 'to_dict'):
            row_dict = latest_row.to_dict()
        else:
            row_dict = dict(latest_row)

        # Pastikan semua nilai JSON-serializable (convert numpy → python native)
        row_clean = {}
        for k, v in row_dict.items():
            try:
                row_clean[str(k)] = float(v) if not isinstance(v, str) else v
            except (TypeError, ValueError):
                row_clean[str(k)] = 0.0

        payload = {
            "ticker": ticker,
            "latest_row": row_clean,
        }

        headers = {
            "Authorization": f"Bearer {HF_API_KEY}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            MODEL_API_URL,
            headers=headers,
            data=json.dumps(payload),
            timeout=30,
        )

        if response.status_code == 200:
            result = response.json()
            pred = float(result.get("predicted_return", 0.0))
            print(f"[API] Prediksi dari HuggingFace API untuk {ticker}: {pred:.4f}")
            return pred
        else:
            print(f"[WARN] HuggingFace API error {response.status_code}: {response.text[:200]}")
            return None

    except requests.exceptions.Timeout:
        print(f"[WARN] HuggingFace API timeout untuk {ticker}. Fallback ke lokal.")
        return None
    except Exception as e:
        print(f"[WARN] HuggingFace API gagal untuk {ticker}: {e}. Fallback ke lokal.")
        return None


def predict_return(ticker, latest_row, stock_df=None, gold_df=None):
    # ── Coba via API GPU (Hugging Face) terlebih dahulu ──
    api_pred = predict_via_api(ticker, latest_row, stock_df, gold_df)
    if api_pred is not None:
        return api_pred, "Machine Learning (GRU+XGBoost via API)"

    # ── Fallback: muat model lokal jika MODEL_API_URL tidak dikonfigurasi ──
    models    = load_ml_models(ticker)
    gru_model = models.get("gru")    if models else None
    xgb_model = models.get("xgb")    if models else None
    scaler    = models.get("scaler") if models else None

    if gru_model and scaler and stock_df is not None and gold_df is not None:
        X_gru = build_gru_window(stock_df, gold_df, scaler, ticker)
        if X_gru is not None:
            from tensorflow import keras
            bottleneck = keras.Model(
                inputs=gru_model.input,
                outputs=gru_model.layers[-2].output
            )
            gru_features = bottleneck.predict(X_gru, verbose=0)
            if xgb_model is not None:
                if gru_features.shape[1] != xgb_model.n_features_in_:
                    raise ValueError(
                        f"XGBoost expected {xgb_model.n_features_in_} features, "
                        f"got {gru_features.shape[1]}"
                    )
                pred = float(xgb_model.predict(gru_features)[0])
                return pred, "Machine Learning (GRU+XGBoost Lokal)"

    print("[WARN] ML tidak tersedia — pakai Rule-Based fallback.")
    return fallback_predict_return(latest_row), "Rule-Based Evaluation"


def fallback_predict_return(row):
    """
    Rule-based fallback.
    Pakai Composite_Rank (0–1) untuk fundamental_signal karena threshold
    di sini disesuaikan dengan skala 0–1.
    (Composite_Rank 0.5 = netral, >0.5 = lebih baik dari median)
    """
    technical_signal = 0
    if row["Close"] > row["MA7"]:  technical_signal += 0.008
    else:                           technical_signal -= 0.004
    if row["MA7"] > row["MA30"]:   technical_signal += 0.012
    else:                           technical_signal -= 0.006

    sentiment_signal   = row["Sentiment_Score"] * 0.012
    # Composite_Rank: 0.5 = netral → (0.5 - 0.5) * 0.025 = 0.0
    fundamental_signal = (row["Composite_Rank"] - 0.5) * 0.025
    gold_signal        = row["Gold_Return"] * 0.6 if not pd.isna(row.get("Gold_Return", float('nan'))) else 0
    volatility_penalty = min(row.get("Volatility", 0) or 0, 0.08) * 0.15

    predicted = technical_signal + sentiment_signal + fundamental_signal + gold_signal - volatility_penalty
    return float(np.clip(predicted, -0.08, 0.12))
