from flask import Flask, render_template, request
from config import TICKER_MAP, COMPANY_NAMES
from utils.data_fetcher import fetch_stock_data, fetch_gold_data, load_fundamental_data, fetch_news, fetch_market_news
from utils.sentiment_analyzer import apply_sentiment, apply_market_sentiment
from utils.model_predictor import predict_return, KAGGLE_PILAR_DATA
from utils.business_logic import prepare_latest_row, generate_recommendation, generate_explanation, calculate_risk_level, detect_overhyped_status, generate_watchlist, get_top_recommendation, compare_stocks

app = Flask(__name__)

@app.route('/')
def dashboard():
    watchlist_df = generate_watchlist()
    top_pick = get_top_recommendation(watchlist_df)
    if top_pick is not None:
        try:
            top_pick = top_pick.to_dict()
        except AttributeError:
            pass
        
        fundamental_df = load_fundamental_data()
        ticker_fund = fundamental_df[fundamental_df["Ticker"] == top_pick.get("Ticker")]
        if not ticker_fund.empty:
            fund_dict = ticker_fund.iloc[0].to_dict()
            top_pick["PE Ratio"] = fund_dict.get("Relative_PE_ratio", 15.0)
        else:
            top_pick["PE Ratio"] = 15.0
    
    market_news_df = fetch_market_news()
    if not market_news_df.empty:
        market_news_df, market_avg_score, market_label = apply_market_sentiment(market_news_df)
    else:
        market_avg_score = 0
        market_label = "Netral"
        
    return render_template('dashboard.html', 
                           top_pick=top_pick, 
                           watchlist=watchlist_df,
                           market_news=market_news_df,
                           market_avg_score=market_avg_score,
                           market_label=market_label)

@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    recommendation = None
    if request.method == 'POST':
        selected_ticker = request.form.get('ticker', 'ANTM')
        selected_goal = request.form.get('goal', 'Jangka Panjang')
        
        fundamental_df = load_fundamental_data()
        ticker_fundamental = fundamental_df[fundamental_df["Ticker"] == selected_ticker]
        fundamental_row = ticker_fundamental.iloc[0].copy() if not ticker_fundamental.empty else None
        
        stock_df = fetch_stock_data(TICKER_MAP[selected_ticker])
        gold_df = fetch_gold_data()
        news_df = fetch_news(selected_ticker)
        news_df, avg_sentiment_score, sentiment_label, news_count = apply_sentiment(news_df)
        
        data = KAGGLE_PILAR_DATA.get(selected_ticker)
        if data:
            avg_sentiment_score = data["sentiment_score"]
            sentiment_label = "Positive" if avg_sentiment_score > 0 else "Negative" if avg_sentiment_score < 0 else "Neutral"
            fundamental_row["Composite_Rank"] = data["piotroski_fuzzy"]
            
        latest_row = prepare_latest_row(stock_df, gold_df, fundamental_row, avg_sentiment_score, news_count)
        predicted_return, model_source = predict_return(selected_ticker, latest_row, stock_df, gold_df)
        
        recom, risk_level, overall_score = generate_recommendation(
            selected_ticker, predicted_return, avg_sentiment_score, fundamental_row["Composite_Rank"], selected_goal
        )
        
        fund_label = "Kuat / Good Fundamental" if fundamental_row["Composite_Rank"] >= 0 else "Lemah / Weak Fundamental"
        explanation = generate_explanation(selected_ticker, recom, predicted_return, sentiment_label, fund_label, selected_goal)
        
        risk_level_str, risk_score, risk_reason = calculate_risk_level(latest_row.get("Volatility", 0), predicted_return, avg_sentiment_score, fundamental_row["Composite_Rank"], selected_goal)
        hype_status, hype_score, hype_reason = detect_overhyped_status(predicted_return, avg_sentiment_score, news_count, fundamental_row["Composite_Rank"])
        
        return render_template('analysis.html',
                               ticker_map=TICKER_MAP,
                               company_names=COMPANY_NAMES,
                               selected_ticker=selected_ticker,
                               selected_goal=selected_goal,
                               recommendation=recom,
                               predicted_return=predicted_return,
                               overall_score=overall_score,
                               sentiment_label=sentiment_label,
                               fundamental_score=fundamental_row["Composite_Rank"],
                               risk_level=risk_level_str,
                               risk_score=risk_score,
                               risk_reason=risk_reason,
                               hype_status=hype_status,
                               hype_score=hype_score,
                               hype_reason=hype_reason,
                               explanation=explanation,
                               latest_row=latest_row.to_dict() if hasattr(latest_row, 'to_dict') else latest_row,
                               avg_score=avg_sentiment_score)
        
    return render_template('analysis.html', ticker_map=TICKER_MAP, company_names=COMPANY_NAMES)

@app.route('/detail')
def detail():
    selected_ticker = request.args.get('ticker', 'ANTM')
    stock_df = fetch_stock_data(TICKER_MAP.get(selected_ticker, "ANTM.JK"))
    fundamental_df = load_fundamental_data()
    ticker_fundamental = fundamental_df[fundamental_df["Ticker"] == selected_ticker]
    fundamental_row = ticker_fundamental.iloc[0].to_dict() if not ticker_fundamental.empty else None
    
    latest_row = stock_df.iloc[-1].to_dict() if not stock_df.empty else {}
    current_price = latest_row.get("Close", 0)
    daily_return = latest_row.get("Return", 0)
    price_change = current_price - (stock_df.iloc[-2]["Close"] if len(stock_df) > 1 else current_price)
    volatility = latest_row.get("Volatility", 0.0)
    
    news_df = fetch_news(selected_ticker)
    if not news_df.empty:
        _, avg_score, label, _ = apply_sentiment(news_df)
    else:
        avg_score, label = 0, "Netral"
    
    chart_html = "<div>Chart will go here.</div>"
    
    return render_template('detail.html', 
                           selected_ticker=selected_ticker, 
                           fundamental_row=fundamental_row, 
                           chart_html=chart_html,
                           current_price=current_price,
                           daily_return=daily_return,
                           price_change=price_change,
                           volatility=volatility,
                           sentiment_score=avg_score,
                           sentiment_label=label)

@app.route('/news')
def news():
    category = request.args.get('category', 'Semua Berita')
    market_news = fetch_market_news(category)
    if not market_news.empty:
        market_news, avg_score, sentiment_label = apply_market_sentiment(market_news)
    else:
        avg_score, sentiment_label = 0, "Netral"
    
    return render_template('news.html', 
                           market_news=market_news, 
                           category=category,
                           avg_score=avg_score,
                           sentiment_label=sentiment_label)

@app.route('/simulator', methods=['GET', 'POST'])
def simulator():
    selected_ticker = request.args.get('ticker', 'ANTM')
    
    stock_df = fetch_stock_data(TICKER_MAP.get(selected_ticker, "ANTM.JK"))
    gold_df = fetch_gold_data()
    fundamental_df = load_fundamental_data()
    
    ticker_fund = fundamental_df[fundamental_df["Ticker"] == selected_ticker]
    fund_dict = ticker_fund.iloc[0].to_dict() if not ticker_fund.empty else {}
    
    news_df = fetch_news(selected_ticker)
    if not news_df.empty:
        _, avg_score, _, _ = apply_sentiment(news_df)
    else:
        avg_score = 0
        
    latest_row = prepare_latest_row(stock_df, gold_df, fund_dict, avg_score, len(news_df))
    current_price = latest_row.get("Close", 1500)
    
    predicted_return, _ = predict_return(selected_ticker, latest_row, stock_df, gold_df)
    
    if predicted_return < -0.80:
        predicted_return = -0.80
    elif predicted_return > 1.00:
        predicted_return = 1.00
    
    recom, _, _ = generate_recommendation(selected_ticker, predicted_return, avg_score, fund_dict.get("Composite_Rank", 0), "Jangka Panjang")
    
    return render_template('simulator.html',
                           selected_ticker=selected_ticker,
                           current_price=current_price,
                           predicted_return=predicted_return,
                           recommendation=recom)

@app.route('/compare')
def compare():
    watchlist_df = generate_watchlist()
    return render_template('compare.html', watchlist=watchlist_df)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, port=5000)
