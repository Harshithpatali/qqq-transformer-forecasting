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
# DEBUG PATHS
# ==========================================

print("\nCURRENT WORKING DIRECTORY:")
print(os.getcwd())

print("\nMODEL PATH:")
print(MODEL_PATH)

print("\nSCALER PATH:")
print(SCALER_PATH)

print("\nMODEL EXISTS:")
print(os.path.exists(MODEL_PATH))

print("\nSCALER EXISTS:")
print(os.path.exists(SCALER_PATH))

# ==========================================
# DEVICE
# ==========================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("\nUsing Device:", DEVICE)

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

print("\nTransformer model loaded.")

# ==========================================
# LOAD SCALER
# ==========================================

scaler = joblib.load(
    SCALER_PATH
)

print("\nScaler loaded.")