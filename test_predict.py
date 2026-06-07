import pandas as pd
from utils.data_fetcher import fetch_stock_data, fetch_gold_data, load_fundamental_data
from utils.model_predictor import predict_return
from utils.business_logic import prepare_latest_row

def test_inference():
    ticker = "ANTM"
    print(f"[*] Testing Pipeline V2 untuk {ticker}...")
    
    # 1. Fetch data pake modul data_fetcher
    stock_df = fetch_stock_data(ticker + ".JK")
    gold_df = fetch_gold_data()
    fund_df = load_fundamental_data()
    
    # Ambil fundamental row
    fund_row = fund_df[fund_df["Ticker"] == ticker].iloc[0].to_dict() if not fund_df.empty else {}
    
    # 2. Persiapan fitur (Wajib pake prepare_latest_row dari business_logic atau model_predictor)
    # Sesuaikan dengan cara app.py manggil data
    latest_row = prepare_latest_row(stock_df, gold_df, fund_row, 0.5, 5)
    
    # 3. Test Prediksi AI
    try:
        pred, source = predict_return(ticker, latest_row, stock_df, gold_df)
        print(f"\n=== HASIL PREDIKSI V2 ===")
        print(f"Saham: {ticker}")
        print(f"Prediksi Return: {pred:.4f}")
        print(f"Model: {source}")
    except Exception as e:
        print(f"\n[!] Error saat inferensi: {e}")

if __name__ == "__main__":
    test_inference()