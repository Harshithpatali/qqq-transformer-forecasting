import os
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
# DEBUGGING
# ==========================================

print("\n========== STARTUP DEBUG ==========")

print("\nCURRENT WORKING DIRECTORY:")
print(os.getcwd())

print("\nROOT DIRECTORY CONTENTS:")
print(os.listdir("/app"))

if os.path.exists("/app/saved_models"):
    print("\nSAVED_MODELS CONTENTS:")
    print(os.listdir("/app/saved_models"))

if os.path.exists("/app/scalers"):
    print("\nSCALERS CONTENTS:")
    print(os.listdir("/app/scalers"))

print("\nMODEL PATH:")
print(MODEL_PATH)

print("\nSCALER PATH:")
print(SCALER_PATH)

print("\nMODEL EXISTS:")
print(os.path.exists(MODEL_PATH))

print("\nSCALER EXISTS:")
print(os.path.exists(SCALER_PATH))

print("\n===================================\n")

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
# LOAD MODEL SAFELY
# ==========================================

try:

    model = TransformerForecaster().to(DEVICE)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.eval()

    print("Transformer model loaded successfully.")

except Exception as e:

    print("\nMODEL LOAD ERROR:")
    print(str(e))

    raise e

# ==========================================
# LOAD SCALER SAFELY
# ==========================================

try:

    scaler = joblib.load(
        SCALER_PATH
    )

    print("Scaler loaded successfully.")

except Exception as e:

    print("\nSCALER LOAD ERROR:")
    print(str(e))

    raise e


# ==========================================
# FETCH MARKET DATA
# ==========================================

def fetch_latest_data():

    df = yf.download(
        TICKER,
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = (
            df.columns.get_level_values(0)
        )

    df.reset_index(inplace=True)

    return df


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(df):

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

    # RETURNS

    df["Returns"] = (
        df["Close"] - df["Close"].shift(1)
    ) / df["Close"].shift(1)

    # LOG RETURNS

    df["Log_Returns"] = np.log(
        df["Close"] / df["Close"].shift(1)
    )

    # ROLLING

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

    # RSI

    rsi = ta.momentum.RSIIndicator(
        close=df["Close"],
        window=14
    )

    df["RSI"] = rsi.rsi()

    # MACD

    macd = ta.trend.MACD(
        close=df["Close"]
    )

    df["MACD"] = macd.macd()

    df["MACD_Signal"] = (
        macd.macd_signal()
    )

    # EMA

    ema20 = ta.trend.EMAIndicator(
        close=df["Close"],
        window=20
    )

    ema50 = ta.trend.EMAIndicator(
        close=df["Close"],
        window=50
    )

    df["EMA20"] = (
        ema20.ema_indicator()
    )

    df["EMA50"] = (
        ema50.ema_indicator()
    )

    # BOLLINGER

    bollinger = (
        ta.volatility.BollingerBands(
            close=df["Close"],
            window=20
        )
    )

    df["BB_High"] = (
        bollinger.bollinger_hband()
    )

    df["BB_Low"] = (
        bollinger.bollinger_lband()
    )

    # ATR

    atr = (
        ta.volatility.AverageTrueRange(
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            window=14
        )
    )

    df["ATR"] = (
        atr.average_true_range()
    )

    df.dropna(inplace=True)

    return df


# ==========================================
# PREPARE SEQUENCE
# ==========================================

def prepare_latest_sequence(df):

    features = df[
        FEATURE_COLUMNS
    ]

    scaled_features = scaler.transform(
        features
    )

    latest_sequence = scaled_features[
        -SEQUENCE_LENGTH:
    ]

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
# SIGNAL
# ==========================================

def generate_signal(predicted_return):

    if predicted_return > 0.002:

        return "BUY"

    elif predicted_return < -0.002:

        return "SELL"

    return "HOLD"


# ==========================================
# CONFIDENCE
# ==========================================

def estimate_confidence(predicted_return):

    confidence = min(
        abs(predicted_return) * 100,
        0.95
    )

    return round(
        float(confidence),
        2
    )


# ==========================================
# MAIN PREDICTION
# ==========================================

def predict_next_day():

    df = fetch_latest_data()

    df = create_features(df)

    current_price = float(
        df["Close"].iloc[-1]
    )

    latest_sequence = (
        prepare_latest_sequence(df)
    )

    with torch.no_grad():

        predicted_return = model(
            latest_sequence
        ).item()

    predicted_price = (
        current_price
        * (1 + predicted_return)
    )

    signal = generate_signal(
        predicted_return
    )

    confidence = estimate_confidence(
        predicted_return
    )

    return {

        "ticker": TICKER,

        "current_price":
            round(current_price, 2),

        "predicted_return":
            round(predicted_return, 6),

        "predicted_price":
            round(predicted_price, 2),

        "signal":
            signal,

        "confidence":
            confidence
    }