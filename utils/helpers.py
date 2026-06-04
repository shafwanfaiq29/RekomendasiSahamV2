import plotly.graph_objects as go

def format_percent(value):
    try:
        return f"{value * 100:.2f}%"
    except Exception:
        return "-"

def format_number(value):
    try:
        if abs(value) >= 1_000_000_000_000:
            return f"{value / 1_000_000_000_000:.2f} T"
        if abs(value) >= 1_000_000_000:
            return f"{value / 1_000_000_000:.2f} B"
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.2f} M"
        return f"{value:,.2f}"
    except Exception:
        return "-"

def create_market_sentiment_chart(news_df):
    if news_df.empty or "Sentiment_Label" not in news_df.columns:
        return None
        
    sentiment_counts = news_df["Sentiment_Label"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    
    colors = {'Positive': '#4ade80', 'Neutral': '#94a3b8', 'Negative': '#f87171'}
    
    fig = go.Figure(data=[go.Pie(
        labels=sentiment_counts["Sentiment"],
        values=sentiment_counts["Count"],
        hole=0.6,
        marker=dict(colors=[colors.get(l, '#94a3b8') for l in sentiment_counts["Sentiment"]])
    )])
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="Distribusi Sentimen Market",
        showlegend=False
    )
    return fig

def create_category_bar_chart(news_df):
    if news_df.empty:
        return None
        
    cat_counts = news_df["Category"].value_counts().reset_index()
    cat_counts.columns = ["Category", "Count"]
    
    fig = go.Figure(go.Bar(
        x=cat_counts["Count"],
        y=cat_counts["Category"],
        orientation='h',
        marker_color="#d4af37"
    ))
    
    fig.update_layout(
        template="plotly_dark",
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0"),
        title="Jumlah Berita per Kategori",
        yaxis={'categoryorder':'total ascending'}
    )
    return fig

def generate_market_insight(sentiment_label, avg_score, top_category):
    if sentiment_label == "Positive Market":
        return "Sentimen pasar saat ini cenderung positif karena mayoritas berita terbaru menunjukkan tren penguatan, prospek pasar, atau kondisi ekonomi yang stabil. Namun, investor tetap perlu memperhatikan risiko volatilitas dan perubahan sentimen global yang bisa terjadi kapan saja."
    elif sentiment_label == "Negative Market":
        return "Sentimen pasar saat ini cenderung negatif karena banyak berita terkait pelemahan harga, tekanan ekonomi, risiko global, atau ketidakpastian pasar. Investor disarankan lebih berhati-hati sebelum mengambil keputusan besar."
    else:
        return "Sentimen pasar saat ini cenderung netral. Berita yang muncul masih sangat beragam antara sentimen positif dan negatif, sehingga investor disarankan untuk tidak hanya mengikuti hype pasar dan tetap memantau tren yang berkembang."
