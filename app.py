import json
from flask import Flask, render_template, request
from config import TICKER_MAP, COMPANY_NAMES
from utils.data_fetcher import (
    fetch_stock_data, fetch_gold_data,
    load_fundamental_data, get_fundamental_row,
    fetch_news, fetch_market_news
)
from utils.sentiment_analyzer import apply_sentiment, apply_market_sentiment
from utils.model_predictor import predict_return, KAGGLE_PILAR_DATA
from utils.business_logic import (
    prepare_latest_row, generate_recommendation, generate_explanation,
    calculate_risk_level, detect_overhyped_status,
    generate_watchlist, get_top_recommendation, compare_stocks
)

app = Flask(__name__)


def _safe_float(val, default=0.0):
    """Konversi nilai ke float aman — tidak pernah crash."""
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


# ==============================================================================
# ROUTE: Dashboard
# ==============================================================================
@app.route('/')
def dashboard():
    watchlist_df = generate_watchlist()
    top_pick = get_top_recommendation(watchlist_df)

    if top_pick is not None:
        try:
            top_pick = top_pick.to_dict()
        except AttributeError:
            pass

        ticker_top = top_pick.get("Ticker", "")
        fund_dict  = get_fundamental_row(ticker_top)

        # Relative_PE_ratio: skala 0.0–1.0 (rasio PE vs benchmark)
        # Tampilkan hanya jika > 0; 0.0 berarti EPS negatif → N/A
        raw_pe = fund_dict.get("Relative_PE_ratio", None)
        top_pick["PE Ratio"] = raw_pe if (raw_pe and raw_pe > 0) else None

    market_news_df = fetch_market_news()
    if not market_news_df.empty:
        market_news_df, market_avg_score, market_label = apply_market_sentiment(market_news_df)
    else:
        market_avg_score, market_label = 0, "Netral"

    return render_template(
        'dashboard.html',
        top_pick=top_pick,
        watchlist=watchlist_df,
        market_news=market_news_df,
        market_avg_score=market_avg_score,
        market_label=market_label,
    )


# ==============================================================================
# ROUTE: Analysis
# ==============================================================================
@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    if request.method == 'POST':
        selected_ticker = request.form.get('ticker', 'ANTM')
        selected_goal   = request.form.get('goal',   'Jangka Panjang')

        fund_dict = get_fundamental_row(selected_ticker)

        # ── Dua nilai fundamental, dua tujuan berbeda ──
        # Skor_Piotroski_Fuzzy (-1/+1) → fuzzy Mamdani, calculate_risk_level, detect_overhyped
        # Composite_Rank (0-1)          → fitur model ML, tampilan skor di template
        piotroski_fuzzy = _safe_float(fund_dict.get("Skor_Piotroski_Fuzzy", 0.0))
        composite_rank  = _safe_float(fund_dict.get("Composite_Rank", 0.0))

        stock_df = fetch_stock_data(TICKER_MAP[selected_ticker])
        gold_df  = fetch_gold_data()
        news_df  = fetch_news(selected_ticker)
        news_df, avg_sentiment_score, sentiment_label, news_count = apply_sentiment(news_df)

        # Override sentimen dari Kaggle jika tersedia (opsional)
        kaggle_data = KAGGLE_PILAR_DATA.get(selected_ticker)
        if kaggle_data and kaggle_data.get("sentiment_score", 0.0) != 0.0:
            avg_sentiment_score = float(kaggle_data["sentiment_score"])
            sentiment_label = (
                "Positive" if avg_sentiment_score > 0
                else "Negative" if avg_sentiment_score < 0
                else "Neutral"
            )

        # prepare_latest_row menggunakan Composite_Rank dari fund_dict untuk fitur ML
        latest_row = prepare_latest_row(
            stock_df, gold_df, fund_dict, avg_sentiment_score, news_count
        )
        predicted_return, model_source = predict_return(
            selected_ticker, latest_row, stock_df, gold_df
        )

        # generate_recommendation menerima piotroski_fuzzy untuk fuzzy Mamdani
        recom, risk_level, overall_score = generate_recommendation(
            selected_ticker, predicted_return, avg_sentiment_score,
            piotroski_fuzzy, selected_goal
        )

        # Label fundamental berdasarkan Skor_Piotroski_Fuzzy
        if piotroski_fuzzy >= 0.33:
            fund_label = "Kuat / Good Fundamental"
        elif piotroski_fuzzy <= -0.33:
            fund_label = "Lemah / Weak Fundamental"
        else:
            fund_label = "Netral / Average Fundamental"

        explanation = generate_explanation(
            selected_ticker, recom, predicted_return, sentiment_label, fund_label, selected_goal
        )

        risk_level_str, risk_score, risk_reason = calculate_risk_level(
            latest_row.get("Volatility", 0), predicted_return,
            avg_sentiment_score, piotroski_fuzzy, selected_goal
        )
        hype_status, hype_score, hype_reason = detect_overhyped_status(
            predicted_return, avg_sentiment_score, news_count, piotroski_fuzzy
        )

        return render_template(
            'analysis.html',
            ticker_map=TICKER_MAP,
            company_names=COMPANY_NAMES,
            selected_ticker=selected_ticker,
            selected_goal=selected_goal,
            recommendation=recom,
            predicted_return=predicted_return,
            overall_score=overall_score,
            sentiment_label=sentiment_label,
            # Template menerima composite_rank (0-1) untuk tampilan skor
            fundamental_score=composite_rank,
            # Dan piotroski_fuzzy (-1/+1) untuk label teks
            piotroski_fuzzy=piotroski_fuzzy,
            risk_level=risk_level_str,
            risk_score=risk_score,
            risk_reason=risk_reason,
            hype_status=hype_status,
            hype_score=hype_score,
            hype_reason=hype_reason,
            explanation=explanation,
            latest_row=latest_row.to_dict() if hasattr(latest_row, 'to_dict') else latest_row,
            avg_score=avg_sentiment_score,
        )

    return render_template('analysis.html', ticker_map=TICKER_MAP, company_names=COMPANY_NAMES)


# ==============================================================================
# ROUTE: Detail
# ==============================================================================
@app.route('/detail')
def detail():
    selected_ticker = request.args.get('ticker', 'ANTM')
    stock_df = fetch_stock_data(TICKER_MAP.get(selected_ticker, "ANTM.JK"))

    fund_dict = get_fundamental_row(selected_ticker)

    latest_row    = stock_df.iloc[-1].to_dict() if not stock_df.empty else {}
    current_price = latest_row.get("Close", 0)
    daily_return  = latest_row.get("Return", 0)
    price_change  = current_price - (
        stock_df.iloc[-2]["Close"] if len(stock_df) > 1 else current_price
    )
    volatility = latest_row.get("Volatility", 0.0)

    news_df = fetch_news(selected_ticker)
    if not news_df.empty:
        _, avg_score, label, _ = apply_sentiment(news_df)
    else:
        avg_score, label = 0, "Netral"

    # Company name dari config (bukan dari CSV fundamental)
    company_name = COMPANY_NAMES.get(selected_ticker, selected_ticker)

    # Bangun data chart untuk Plotly: Close, MA7, MA30
    def _safe_val(v):
        import math
        if v is None: return None
        try:
            f = float(v)
            return None if math.isnan(f) or math.isinf(f) else round(f, 2)
        except Exception:
            return None

    stock_chart_json = json.dumps({
        "ticker": selected_ticker,
        "dates":  [str(d.date()) if hasattr(d, 'date') else str(d)
                   for d in stock_df["Date"].tolist()],
        "close":  [_safe_val(v) for v in stock_df["Close"].tolist()],
        "ma7":    [_safe_val(v) for v in stock_df["MA7"].tolist()],
        "ma30":   [_safe_val(v) for v in stock_df["MA30"].tolist()],
    })

    return render_template(
        'detail.html',
        selected_ticker=selected_ticker,
        company_name=company_name,
        fundamental_row=fund_dict,
        stock_chart_json=stock_chart_json,
        current_price=current_price,
        daily_return=daily_return,
        price_change=price_change,
        volatility=volatility,
        sentiment_score=avg_score,
        sentiment_label=label,
    )


# ==============================================================================
# ROUTE: News
# ==============================================================================
@app.route('/news')
def news():
    category    = request.args.get('category', 'Semua Berita')
    market_news = fetch_market_news(category)

    if not market_news.empty:
        market_news, avg_score, sentiment_label = apply_market_sentiment(market_news)
    else:
        avg_score, sentiment_label = 0, "Netral"

    return render_template(
        'news.html',
        market_news=market_news,
        category=category,
        avg_score=avg_score,
        sentiment_label=sentiment_label,
    )


# ==============================================================================
# ROUTE: Simulator
# ==============================================================================
@app.route('/simulator', methods=['GET', 'POST'])
def simulator():
    selected_ticker = request.args.get('ticker', 'ANTM')

    stock_df  = fetch_stock_data(TICKER_MAP.get(selected_ticker, "ANTM.JK"))
    gold_df   = fetch_gold_data()
    fund_dict = get_fundamental_row(selected_ticker)

    news_df = fetch_news(selected_ticker)
    if not news_df.empty:
        _, avg_score, _, _ = apply_sentiment(news_df)
    else:
        avg_score = 0

    latest_row    = prepare_latest_row(stock_df, gold_df, fund_dict, avg_score, len(news_df))
    current_price = latest_row.get("Close", 1500)

    predicted_return, _ = predict_return(selected_ticker, latest_row, stock_df, gold_df)
    predicted_return     = max(-0.80, min(1.00, predicted_return))

    piotroski_fuzzy = _safe_float(fund_dict.get("Skor_Piotroski_Fuzzy", 0.0))
    recom, _, _     = generate_recommendation(
        selected_ticker, predicted_return, avg_score, piotroski_fuzzy, "Jangka Panjang"
    )

    return render_template(
        'simulator.html',
        selected_ticker=selected_ticker,
        current_price=current_price,
        predicted_return=predicted_return,
        recommendation=recom,
    )


# ==============================================================================
# ROUTE: Compare
# ==============================================================================
@app.route('/compare')
def compare():
    watchlist_df = generate_watchlist()
    return render_template('compare.html', watchlist=watchlist_df)


if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)