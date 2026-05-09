import numpy as np
import pandas as pd
import yfinance as yf
import ta
import joblib

import torch

from model import TransformerForecaster

from config import (
    TICKER,
    START_DATE,
    FEATURE_COLUMNS,
    SEQUENCE_LENGTH,
    MODEL_PATH,
    SCALER_PATH
)

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using Device:", DEVICE)

# ==========================================
# LOAD MODEL
# ==========================================

model = TransformerForecaster().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Transformer model loaded.")

# ==========================================
# LOAD SCALER
# ==========================================

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded.")


# ==========================================
# DOWNLOAD LATEST DATA
# ==========================================

def fetch_latest_data():

    print("Fetching latest QQQ data...")

    df = yf.download(
        TICKER,
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )

    # ======================================
    # FIX MULTIINDEX
    # ======================================

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    return df


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(df):

    # ======================================
    # FORCE NUMERIC TYPES
    # ======================================

    numeric_cols = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]

    for col in numeric_cols:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # ======================================
    # RETURNS
    # ======================================

    df["Returns"] = (
        df["Close"] - df["Close"].shift(1)
    ) / df["Close"].shift(1)

    # ======================================
    # LOG RETURNS
    # ======================================

    df["Log_Returns"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # ======================================
    # ROLLING FEATURES
    # ======================================

    df["Rolling_Mean_5"] = (
        df["Close"]
        .rolling(window=5)
        .mean()
    )

    df["Rolling_Std_5"] = (
        df["Close"]
        .rolling(window=5)
        .std()
    )

    df["Rolling_Volatility_10"] = (
        df["Returns"]
        .rolling(window=10)
        .std()
    )

    # ======================================
    # RSI
    # ======================================

    rsi = ta.momentum.RSIIndicator(
        close=df["Close"],
        window=14
    )

    df["RSI"] = rsi.rsi()

    # ======================================
    # MACD
    # ======================================

    macd = ta.trend.MACD(
        close=df["Close"]
    )

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = macd.macd_signal()

    # ======================================
    # EMA
    # ======================================

    ema20 = ta.trend.EMAIndicator(
        close=df["Close"],
        window=20
    )

    ema50 = ta.trend.EMAIndicator(
        close=df["Close"],
        window=50
    )

    df["EMA20"] = ema20.ema_indicator()

    df["EMA50"] = ema50.ema_indicator()

    # ======================================
    # BOLLINGER
    # ======================================

    bollinger = ta.volatility.BollingerBands(
        close=df["Close"],
        window=20
    )

    df["BB_High"] = bollinger.bollinger_hband()

    df["BB_Low"] = bollinger.bollinger_lband()

    # ======================================
    # ATR
    # ======================================

    atr = ta.volatility.AverageTrueRange(
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        window=14
    )

    df["ATR"] = atr.average_true_range()

    # ======================================
    # REMOVE NaN
    # ======================================

    df.dropna(inplace=True)

    return df


# ==========================================
# PREPARE LATEST SEQUENCE
# ==========================================

def prepare_latest_sequence(df):

    # ======================================
    # SELECT FEATURES
    # ======================================

    features = df[
        FEATURE_COLUMNS
    ]

    # ======================================
    # SCALE FEATURES
    # ======================================

    scaled_features = scaler.transform(
        features
    )

    # ======================================
    # GET LAST 60 DAYS
    # ======================================

    latest_sequence = scaled_features[
        -SEQUENCE_LENGTH:
    ]

    # ======================================
    # CONVERT TO TENSOR
    # ======================================

    latest_sequence = np.expand_dims(
        latest_sequence,
        axis=0
    )

    latest_sequence = torch.tensor(
        latest_sequence,
        dtype=torch.float32
    ).to(DEVICE)

    return latest_sequence


# ==========================================
# GENERATE SIGNAL
# ==========================================

def generate_signal(predicted_return):

    if predicted_return > 0.002:

        return "BUY"

    elif predicted_return < -0.002:

        return "SELL"

    else:

        return "HOLD"


# ==========================================
# CONFIDENCE ESTIMATION
# ==========================================

def estimate_confidence(predicted_return):

    confidence = min(
        abs(predicted_return) * 100,
        0.95
    )

    confidence = round(
        float(confidence),
        2
    )

    return confidence


# ==========================================
# MAIN PREDICTION FUNCTION
# ==========================================

def predict_next_day():

    # ======================================
    # FETCH DATA
    # ======================================

    df = fetch_latest_data()

    # ======================================
    # CREATE FEATURES
    # ======================================

    df = create_features(df)

    # ======================================
    # CURRENT PRICE
    # ======================================

    current_price = float(
        df["Close"].iloc[-1]
    )

    # ======================================
    # PREPARE SEQUENCE
    # ======================================

    latest_sequence = (
        prepare_latest_sequence(df)
    )

    # ======================================
    # MODEL PREDICTION
    # ======================================

    with torch.no_grad():

        predicted_return = model(
            latest_sequence
        ).item()

    # ======================================
    # PREDICTED PRICE
    # ======================================

    predicted_price = (
        current_price
        * (1 + predicted_return)
    )

    # ======================================
    # SIGNAL
    # ======================================

    signal = generate_signal(
        predicted_return
    )

    # ======================================
    # CONFIDENCE
    # ======================================

    confidence = estimate_confidence(
        predicted_return
    )

    # ======================================
    # RESULTS
    # ======================================

    results = {

        "ticker": TICKER,

        "current_price": round(
            current_price,
            2
        ),

        "predicted_return": round(
            predicted_return,
            6
        ),

        "predicted_price": round(
            predicted_price,
            2
        ),

        "signal": signal,

        "confidence": confidence
    }

    return results


# ==========================================
# MAIN TEST
# ==========================================

if __name__ == "__main__":

    prediction = predict_next_day()

    print("\n================================")
    print("AI FORECAST")
    print("================================\n")

    for key, value in prediction.items():

        print(f"{key}: {value}")