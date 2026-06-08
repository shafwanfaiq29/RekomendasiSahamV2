"""
test_predict.py — Cek alur fitur utama: data → prepare_latest_row → predict_return

Jalankan dari root project:
    python test_predict.py

Tidak butuh koneksi internet (fundamental dari Excel, ML dari models/).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from utils.data_fetcher import fetch_stock_data, fetch_gold_data, get_fundamental_row
from utils.model_predictor import predict_return
from utils.business_logic import prepare_latest_row


def test_inference(ticker: str = "ANTM"):
    print(f"\n{'='*50}")
    print(f"[*] Testing pipeline untuk: {ticker}")
    print('='*50)

    # 1. Data harga
    stock_df = fetch_stock_data(ticker + ".JK")
    gold_df  = fetch_gold_data()
    print(f"[OK] stock_df  : {len(stock_df)} baris")
    print(f"[OK] gold_df   : {len(gold_df)} baris")

    # 2. Fundamental — pakai get_fundamental_row (single source of truth)
    fund_dict = get_fundamental_row(ticker)
    print(f"[OK] fund_dict : {len(fund_dict)} kolom")
    print(f"     Composite_Rank      = {fund_dict.get('Composite_Rank'):.4f}  (skala 0-1, untuk ML)")
    print(f"     Skor_Piotroski_Fuzzy= {fund_dict.get('Skor_Piotroski_Fuzzy'):+.4f}  (skala -1/+1, untuk Fuzzy Mamdani)")
    print(f"     Harga_Wajar_Graham  = {fund_dict.get('Harga_Wajar_Graham'):.0f}")

    # 3. Feature row
    latest_row = prepare_latest_row(stock_df, gold_df, fund_dict,
                                    sentiment_score=0.5, news_count=5)
    print(f"[OK] latest_row: {len(latest_row)} fitur")
    print(f"     Close           = {latest_row.get('Close'):.0f}")
    print(f"     Composite_Rank  = {latest_row.get('Composite_Rank'):.4f}")

    # 4. Prediksi
    try:
        pred, source = predict_return(ticker, latest_row, stock_df, gold_df)
        print(f"\n=== HASIL PREDIKSI ===")
        print(f"Saham          : {ticker}")
        print(f"Predicted Return: {pred:.4f} ({pred*100:.2f}%)")
        print(f"Model Source   : {source}")
    except Exception as e:
        print(f"\n[!] Error prediksi: {e}")

    # 5. Verifikasi integritas: tidak ada None/NaN di kolom kritis
    critical_cols = [
        "Close", "MA7", "MA30", "Volatility",
        "Composite_Rank", "Skor_Piotroski_Fuzzy",
        "Sentiment_Score", "Gold_Return"
    ]
    print(f"\n=== CEK INTEGRITAS KOLOM KRITIS ===")
    all_ok = True
    for col in critical_cols:
        val = latest_row.get(col)
        import math
        is_bad = val is None or (isinstance(val, float) and math.isnan(val))
        status = "✅" if not is_bad else "❌ NULL/NaN"
        if is_bad:
            all_ok = False
        print(f"  {col:<30}: {val}  {status}")

    print(f"\n{'[PASS] Semua kolom kritis valid.' if all_ok else '[FAIL] Ada kolom bermasalah!'}")


if __name__ == "__main__":
    tickers = sys.argv[1:] if len(sys.argv) > 1 else ["ANTM", "MDKA", "BRMS"]
    for t in tickers:
        test_inference(t)