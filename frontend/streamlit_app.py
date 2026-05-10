import streamlit as st
import requests
import pandas as pd
import yfinance as yf

import plotly.graph_objects as go

from plotly.subplots import make_subplots

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(

    page_title="QQQ Quant AI",

    page_icon="📈",

    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .stMetric {
        background-color: #1A1F2B;
        padding: 15px;
        border-radius: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================
# TITLE
# ==========================================

st.title("📈 AI-Powered QQQ Forecasting Platform")

st.markdown(
    "### Transformer-Based Quant Forecasting System"
)

# ==========================================
# SIDEBAR
# ==========================================

st.sidebar.title("⚙️ Control Panel")

refresh = st.sidebar.button(
    "🔄 Refresh Prediction"
)

auto_refresh = st.sidebar.checkbox(
    "Auto Refresh",
    value=False
)

# ==========================================
# BACKEND API URL
# ==========================================

API_URL = (
    "https://qqq-transformer-forecasting.onrender.com/predict"
)

# ==========================================
# FETCH PREDICTION
# ==========================================

def fetch_prediction():

    try:

        response = requests.post(API_URL)

        data = response.json()

        return data

    except Exception as e:

        st.error(
            f"API Error: {e}"
        )

        return None

# ==========================================
# FETCH MARKET DATA
# ==========================================

@st.cache_data(ttl=60)
def load_market_data():

    df = yf.download(

        "QQQ",

        period="6mo",

        interval="1d",

        auto_adjust=True,

        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = (
            df.columns.get_level_values(0)
        )

    return df

# ==========================================
# LOAD DATA
# ==========================================

market_data = load_market_data()

prediction_data = fetch_prediction()

# ==========================================
# CURRENT PRICE
# ==========================================

current_price = float(
    market_data["Close"].iloc[-1]
)

previous_price = float(
    market_data["Close"].iloc[-2]
)

price_change = (
    current_price - previous_price
)

price_change_pct = (
    price_change / previous_price
) * 100

# ==========================================
# TOP METRICS
# ==========================================

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(

        "QQQ Current Price",

        f"${current_price:.2f}",

        f"{price_change_pct:.2f}%"
    )

with col2:

    if prediction_data:

        predicted_price = (
            prediction_data["data"]
            ["predicted_price"]
        )

        predicted_return = (
            prediction_data["data"]
            ["predicted_return"]
        )

        st.metric(

            "Predicted Price",

            f"${predicted_price:.2f}",

            f"{predicted_return*100:.2f}%"
        )

with col3:

    if prediction_data:

        signal = (
            prediction_data["data"]
            ["signal"]
        )

        st.metric(

            "AI Signal",

            signal
        )

with col4:

    if prediction_data:

        confidence = (
            prediction_data["data"]
            ["confidence"]
        )

        st.metric(

            "Confidence",

            f"{confidence:.2f}"
        )

# ==========================================
# CANDLESTICK CHART
# ==========================================

st.subheader("📊 QQQ Candlestick Chart")

fig = go.Figure()

fig.add_trace(

    go.Candlestick(

        x=market_data.index,

        open=market_data["Open"],

        high=market_data["High"],

        low=market_data["Low"],

        close=market_data["Close"],

        name="QQQ"
    )
)

fig.update_layout(

    template="plotly_dark",

    height=600,

    xaxis_rangeslider_visible=False
)

st.plotly_chart(
    fig,
    use_container_width=True
)

# ==========================================
# MOVING AVERAGES
# ==========================================

st.subheader("📈 Trend Analysis")

market_data["EMA20"] = (
    market_data["Close"]
    .ewm(span=20)
    .mean()
)

market_data["EMA50"] = (
    market_data["Close"]
    .ewm(span=50)
    .mean()
)

fig2 = go.Figure()

fig2.add_trace(

    go.Scatter(

        x=market_data.index,

        y=market_data["Close"],

        mode="lines",

        name="Close"
    )
)

fig2.add_trace(

    go.Scatter(

        x=market_data.index,

        y=market_data["EMA20"],

        mode="lines",

        name="EMA20"
    )
)

fig2.add_trace(

    go.Scatter(

        x=market_data.index,

        y=market_data["EMA50"],

        mode="lines",

        name="EMA50"
    )
)

fig2.update_layout(

    template="plotly_dark",

    height=500
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# ==========================================
# VOLATILITY ANALYSIS
# ==========================================

st.subheader("⚡ Volatility Analysis")

market_data["Returns"] = (
    market_data["Close"]
    .pct_change()
)

market_data["Volatility"] = (
    market_data["Returns"]
    .rolling(10)
    .std()
)

fig3 = go.Figure()

fig3.add_trace(

    go.Scatter(

        x=market_data.index,

        y=market_data["Volatility"],

        mode="lines",

        name="Volatility"
    )
)

fig3.update_layout(

    template="plotly_dark",

    height=400
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

# ==========================================
# AI PREDICTION PANEL
# ==========================================

st.subheader("🤖 AI Prediction Engine")

if prediction_data:

    data = prediction_data["data"]

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("### Prediction Details")

        st.write(
            f"Ticker: {data['ticker']}"
        )

        st.write(
            f"Current Price: "
            f"${data['current_price']}"
        )

        st.write(
            f"Predicted Return: "
            f"{data['predicted_return']*100:.4f}%"
        )

        st.write(
            f"Predicted Price: "
            f"${data['predicted_price']}"
        )

    with col2:

        st.markdown("### Trading Signal")

        signal = data["signal"]

        if signal == "BUY":

            st.success(
                f"BUY SIGNAL ✅"
            )

        elif signal == "SELL":

            st.error(
                f"SELL SIGNAL ❌"
            )

        else:

            st.warning(
                f"HOLD SIGNAL ⚠️"
            )

        st.progress(
            min(
                data["confidence"],
                1.0
            )
        )

        st.write(
            f"Confidence: "
            f"{data['confidence']}"
        )

# ==========================================
# MARKET STATISTICS
# ==========================================

st.subheader("📌 Market Statistics")

stats_col1, stats_col2, stats_col3 = (
    st.columns(3)
)

with stats_col1:

    st.metric(

        "52W High",

        f"${market_data['High'].max():.2f}"
    )

with stats_col2:

    st.metric(

        "52W Low",

        f"${market_data['Low'].min():.2f}"
    )

with stats_col3:

    avg_volume = (
        market_data["Volume"]
        .mean()
    )

    st.metric(

        "Average Volume",

        f"{avg_volume:,.0f}"
    )

# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.markdown(
    "### 🚀 Institutional Quant Dashboard"
)

st.markdown(
    "Powered by Transformer Neural Networks "
    "and FastAPI"
)