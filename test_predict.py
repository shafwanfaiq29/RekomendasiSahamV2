import os
import pandas as pd

from app import predict_return, ambil_data_asli_kaggle, fetch_gold_data_local

def test_inference():
    ticker = "ANTM"
    print(f"Testing inference for {ticker}...")
    
    # 1. Load data
    stock_df, latest_row = ambil_data_asli_kaggle(ticker)
    gold_df = fetch_gold_data_local()
    
    if stock_df is None or gold_df is None:
        print("Failed to load data")
        return
        
    print(f"Stock Data Rows: {len(stock_df)}")
    print(f"Gold Data Rows: {len(gold_df)}")
    
    # 2. Predict return
    try:
        pred, source = predict_return(ticker, latest_row, stock_df, gold_df)
        print("\n=== FINAL TEST RESULT ===")
        print(f"Prediction: {pred}")
        print(f"Source: {source}")
    except Exception as e:
        print("\n=== EXCEPTION CAUGHT ===")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_inference()
