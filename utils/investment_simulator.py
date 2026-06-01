import pandas as pd
import plotly.graph_objects as go


def format_idr(value: float) -> str:
    """Format nilai ke Rupiah Indonesia."""
    try:
        amount = int(round(value, 0))
        return f"Rp {amount:,}".replace(",", ".")
    except Exception:
        return "Rp 0"


def simulate_investment_return(
    investment_amount: float,
    current_price: float,
    ticker: str,
    investment_goal: str,
    duration: int,
    predicted_return: float = 0.05,
) -> dict:
    """
    Simulasikan return investasi berdasarkan nominal, harga saham, ticker, tujuan, dan durasi.
    """
    pred_return = float(predicted_return) if predicted_return is not None else 0.05
    pred_return = max(-0.99, min(pred_return, 5.0))   # guard extreme values

    # 1. Jumlah saham terbeli
    shares_bought = investment_amount / current_price if current_price > 0 else 0

    if investment_goal == "Jangka Pendek":
        # Konversi annual return ke monthly return
        monthly_return = ((1 + pred_return) ** (1 / 12)) - 1 if pred_return > -1 else 0.0
        
        # Estimasi harga masa depan
        future_price = current_price * ((1 + monthly_return) ** duration)
        
        # Nilai investasi akhir
        future_value = shares_bought * future_price
        
        annualized_return = (
            ((future_value / investment_amount) ** (12 / duration) - 1)
            if duration > 0 and investment_amount > 0
            else 0.0
        )
        period_unit = "Bulan"
        period_label = "Bulan ke-"

        curve_points = [investment_amount]
        curve_labels = [0]
        for month in range(1, duration + 1):
            price_at_month = current_price * ((1 + monthly_return) ** month)
            curve_points.append(shares_bought * price_at_month)
            curve_labels.append(month)
    else:
        # Jangka Panjang
        years = duration
        
        # Estimasi harga masa depan
        future_price = current_price * ((1 + pred_return) ** years)
        
        # Nilai investasi akhir
        future_value = shares_bought * future_price
        
        annualized_return = (
            ((future_value / investment_amount) ** (1 / years) - 1)
            if years > 0 and investment_amount > 0
            else 0.0
        )
        period_unit = "Tahun"
        period_label = "Tahun ke-"

        curve_points = [investment_amount]
        curve_labels = [0]
        for year in range(1, years + 1):
            price_at_year = current_price * ((1 + pred_return) ** year)
            curve_points.append(shares_bought * price_at_year)
            curve_labels.append(year)

    # 4. Profit / Loss
    profit = future_value - investment_amount
    
    # 5. Persentase gain
    percent_gain = ((future_value - investment_amount) / investment_amount) * 100 if investment_amount > 0 else 0

    # Simple risk classification from annualized return
    abs_return = abs(pred_return)
    if abs_return < 0.05:
        risk_level = "LOW RISK"
    elif abs_return < 0.20:
        risk_level = "MODERATE RISK"
    else:
        risk_level = "HIGH RISK"

    curve_df = pd.DataFrame({"Period": curve_labels, "Value": curve_points})

    fig = go.Figure(
        go.Scatter(
            x=curve_df["Period"],
            y=curve_df["Value"],
            mode="lines+markers",
            line=dict(color="#f7d774", width=3),
            marker=dict(color="#f7d774", size=8),
            hovertemplate="%{y:,.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(title="Nilai Investasi (Rp)", tickformat=",.0f"),
        xaxis=dict(title=period_label),
        font=dict(color="#e2e8f0"),
        height=350,
    )

    insight = (
        f"Jika Anda berinvestasi {format_idr(investment_amount)} pada {ticker} "
        f"selama {duration} {period_unit.lower()}, berdasarkan prediksi model saat ini "
        f"nilai investasi berpotensi menjadi {format_idr(future_value)}, "
        f"dengan estimasi keuntungan {format_idr(profit)}."
    )

    return {
        "ticker": ticker,
        "investment_amount": investment_amount,
        "future_value": future_value,
        "profit": profit,
        "percent_gain": percent_gain,
        "annualized_return": annualized_return,
        "period_unit": period_unit,
        "risk_level": risk_level,
        "chart_fig": fig,
        "insight": insight,
        "formatted": {
            "investment_amount": format_idr(investment_amount),
            "current_price": f"{format_idr(current_price)} / lembar",
            "estimated_shares": f"{shares_bought:,.2f} lembar".replace(",", "X").replace(".", ",").replace("X", "."),
            "future_price": format_idr(future_price),
            "future_value": format_idr(future_value),
            "profit": f"+{format_idr(profit)}" if profit > 0 else format_idr(profit),
            "percent_gain": f"{percent_gain:+.2f}%",
            "annualized_return": f"{annualized_return * 100:.2f}% / tahun",
        },
    }
