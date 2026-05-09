import pandas as pd
import numpy as np
import yfinance as yf
import ta
import joblib

from sklearn.preprocessing import StandardScaler

from config import (
    TICKER,
    START_DATE,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    SEQUENCE_LENGTH,
    TRAIN_SPLIT,
    VALID_SPLIT,
    DATA_PATH,
    SCALER_PATH
)

print("PREPROCESS PIPELINE STARTED")


# ==========================================
# DOWNLOAD MARKET DATA
# ==========================================

def download_data():

    print("\nDownloading QQQ market data...")

    df = yf.download(
        TICKER,
        start=START_DATE,
        auto_adjust=True,
        progress=False
    )

    # ======================================
    # FIX MULTIINDEX COLUMNS
    # ======================================

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    df.reset_index(inplace=True)

    print("\nDownload complete.")

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns)

    print("\nHead:")
    print(df.head())

    return df


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(df):

    print("\nCreating technical indicators...")

    print("\nInitial Shape:")
    print(df.shape)

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
    # BOLLINGER BANDS
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
    # TARGET VARIABLE
    # ======================================

    df[TARGET_COLUMN] = (
        df["Close"].shift(-1) - df["Close"]
    ) / df["Close"]

    # ======================================
    # CHECK NaN VALUES
    # ======================================

    print("\nNaN Counts Before Drop:")

    print(df.isna().sum())

    # ======================================
    # REMOVE NaN
    # ======================================

    df.dropna(inplace=True)

    # ======================================
    # FINAL SAFETY CHECK
    # ======================================

    if len(df) == 0:

        raise ValueError(
            "Dataset became empty after feature engineering."
        )

    print("\nFeature engineering complete.")

    print("\nFinal Shape:")
    print(df.shape)

    return df


# ==========================================
# TRAIN / VALID / TEST SPLIT
# ==========================================

def split_data(df):

    print("\nSplitting dataset...")

    total_size = len(df)

    train_size = int(total_size * TRAIN_SPLIT)

    valid_size = int(total_size * VALID_SPLIT)

    train_df = df[:train_size]

    valid_df = df[
        train_size:train_size + valid_size
    ]

    test_df = df[
        train_size + valid_size:
    ]

    print("\nTrain Shape:", train_df.shape)
    print("Validation Shape:", valid_df.shape)
    print("Test Shape:", test_df.shape)

    return train_df, valid_df, test_df


# ==========================================
# FEATURE SCALING
# ==========================================

def scale_features(
    train_df,
    valid_df,
    test_df
):

    print("\nScaling features...")

    scaler = StandardScaler()

    # ======================================
    # FIT ONLY ON TRAIN DATA
    # ======================================

    train_scaled = scaler.fit_transform(
        train_df[FEATURE_COLUMNS]
    )

    valid_scaled = scaler.transform(
        valid_df[FEATURE_COLUMNS]
    )

    test_scaled = scaler.transform(
        test_df[FEATURE_COLUMNS]
    )

    # ======================================
    # SAVE SCALER
    # ======================================

    joblib.dump(
        scaler,
        SCALER_PATH
    )

    print("\nScaler saved:")
    print(SCALER_PATH)

    return (
        train_scaled,
        valid_scaled,
        test_scaled,
        scaler
    )


# ==========================================
# CREATE SEQUENCES
# ==========================================

def create_sequences(
    features,
    targets,
    sequence_length
):

    X = []
    y = []

    print("\nCreating sequences...")

    print("Feature Shape:", features.shape)
    print("Target Shape:", targets.shape)

    for i in range(sequence_length, len(features)):

        X.append(
            features[
                i-sequence_length:i
            ]
        )

        y.append(targets[i])

    X = np.array(X)

    y = np.array(y)

    if len(X) == 0:

        raise ValueError(
            "No sequences created."
        )

    print("\nSequence Shape:")
    print(X.shape)

    return X, y


# ==========================================
# COMPLETE DATA PIPELINE
# ==========================================

def prepare_datasets():

    # ======================================
    # DOWNLOAD DATA
    # ======================================

    df = download_data()

    # ======================================
    # CREATE FEATURES
    # ======================================

    df = create_features(df)

    # ======================================
    # SAVE DATASET
    # ======================================

    df.to_csv(DATA_PATH, index=False)

    print("\nDataset saved:")
    print(DATA_PATH)

    # ======================================
    # SPLIT DATA
    # ======================================

    train_df, valid_df, test_df = split_data(df)

    # ======================================
    # SCALE FEATURES
    # ======================================

    train_scaled, valid_scaled, test_scaled, scaler = (
        scale_features(
            train_df,
            valid_df,
            test_df
        )
    )

    # ======================================
    # TARGETS
    # ======================================

    train_targets = train_df[
        TARGET_COLUMN
    ].values

    valid_targets = valid_df[
        TARGET_COLUMN
    ].values

    test_targets = test_df[
        TARGET_COLUMN
    ].values

    # ======================================
    # CREATE SEQUENCES
    # ======================================

    X_train, y_train = create_sequences(
        train_scaled,
        train_targets,
        SEQUENCE_LENGTH
    )

    X_valid, y_valid = create_sequences(
        valid_scaled,
        valid_targets,
        SEQUENCE_LENGTH
    )

    X_test, y_test = create_sequences(
        test_scaled,
        test_targets,
        SEQUENCE_LENGTH
    )

    print("\n===================================")
    print("FINAL DATASET SHAPES")
    print("===================================")

    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)

    print("X_valid:", X_valid.shape)
    print("y_valid:", y_valid.shape)

    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    return (
        X_train,
        y_train,
        X_valid,
        y_valid,
        X_test,
        y_test,
        scaler,
        df
    )


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    print("\nMAIN EXECUTION STARTED")

    prepare_datasets()

    print("\nPIPELINE EXECUTION COMPLETED")