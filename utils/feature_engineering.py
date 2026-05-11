import pandas as pd
import numpy as np


def calculate_technical_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung fitur teknikal dari data harga saham.

    Args:
        stock_df: DataFrame dengan kolom Open, High, Low, Close, Volume

    Returns:
        DataFrame dengan tambahan kolom: Return, MA7, MA30, Volatility
    """
    df = stock_df.copy()
    df.sort_values("Date", inplace=True)

    df["Return"] = df["Close"].pct_change()
    df["MA7"] = df["Close"].rolling(window=7, min_periods=1).mean()
    df["MA30"] = df["Close"].rolling(window=30, min_periods=1).mean()
    df["Volatility"] = df["Return"].rolling(window=7, min_periods=1).std()

    return df


def merge_gold_features(stock_df: pd.DataFrame, gold_df: pd.DataFrame) -> pd.DataFrame:
    """
    Menggabungkan data harga saham dengan data harga emas.

    Args:
        stock_df: DataFrame saham dengan kolom Date
        gold_df: DataFrame emas dengan kolom Date, Gold_Close, Gold_Return

    Returns:
        DataFrame gabungan. Jika gold_df kosong, kolom emas diisi 0.
    """
    if gold_df.empty:
        stock_df["Gold_Close"] = 0.0
        stock_df["Gold_Return"] = 0.0
        return stock_df

    # Pastikan tipe tanggal sama (date only, tanpa timezone)
    stock_df = stock_df.copy()
    gold_df = gold_df.copy()

    stock_df["Date"] = pd.to_datetime(stock_df["Date"]).dt.date
    gold_df["Date"] = pd.to_datetime(gold_df["Date"]).dt.date

    merged = pd.merge(stock_df, gold_df[["Date", "Gold_Close", "Gold_Return"]],
                      on="Date", how="left")
    merged["Gold_Close"] = merged["Gold_Close"].ffill().fillna(0)
    merged["Gold_Return"] = merged["Gold_Return"].ffill().fillna(0.0)
    return merged


def build_feature_row(stock_df: pd.DataFrame, gold_df: pd.DataFrame,
                      sentiment: dict, fundamental: dict) -> pd.DataFrame:
    """
    Membangun satu baris fitur untuk prediksi model (baris terakhir data).

    Args:
        stock_df: DataFrame saham (sudah di-merge gold)
        gold_df: DataFrame emas
        sentiment: dict dari analyze_sentiment()
        fundamental: dict data fundamental ticker

    Returns:
        DataFrame satu baris berisi semua fitur model.
        Kembalikan None jika data tidak cukup.
    """
    # Hitung fitur teknikal
    df = calculate_technical_features(stock_df)
    df = merge_gold_features(df, gold_df)
    df.dropna(subset=["Return"], inplace=True)

    if df.empty:
        return None

    # Ambil baris terakhir (data terbaru)
    last = df.iloc[-1].copy()

    # Gabungkan dengan fitur sentimen
    last["Sentiment_Score"] = sentiment.get("Sentiment_Score", 0.0)
    last["News_Count"] = sentiment.get("News_Count", 0)

    # Gabungkan dengan fitur fundamental
    fundamental_cols = ["PBV_x_ROE", "Price_to_Equity_Discount", "Relative_PE_ratio", "EPS_Growth", "Debt_to_Total_Assets_Ratio", "Liquidity_Differential", "CCE", "Operating_Efficiency", "Dividend_Payout", "Yearly_Price_Change", "Composite_Rank", "Net_Debt_to_Equity"]
    for col in fundamental_cols:
        last[col] = fundamental.get(col, 0.0)

    # Kolom fitur yang dibutuhkan model
    FEATURE_COLS = [
        "Open", "High", "Low", "Close", "Volume",
        "Return", "MA7", "MA30", "Volatility",
        "Gold_Close", "Gold_Return",
        "Sentiment_Score", "News_Count",
        "PBV_x_ROE", "Price_to_Equity_Discount", "Relative_PE_ratio", 
        "EPS_Growth", "Debt_to_Total_Assets_Ratio", "Liquidity_Differential", 
        "CCE", "Operating_Efficiency", "Dividend_Payout", 
        "Yearly_Price_Change", "Composite_Rank", "Net_Debt_to_Equity"
    ]

    feature_row = pd.DataFrame([last[FEATURE_COLS].values], columns=FEATURE_COLS)
    feature_row = feature_row.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return feature_row
