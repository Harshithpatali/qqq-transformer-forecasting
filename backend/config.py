import os

# ==========================================
# BASE DIRECTORY
# ==========================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

print("\nBASE DIRECTORY:")
print(BASE_DIR)

# ==========================================
# MARKET CONFIG
# ==========================================

TICKER = "QQQ"

START_DATE = "2000-01-01"

SEQUENCE_LENGTH = 60

# ==========================================
# FEATURE COLUMNS
# ==========================================

FEATURE_COLUMNS = [

    "Open",
    "High",
    "Low",
    "Close",
    "Volume",

    "Returns",
    "Log_Returns",

    "Rolling_Mean_5",
    "Rolling_Std_5",
    "Rolling_Volatility_10",

    "RSI",

    "MACD",
    "MACD_Signal",

    "EMA20",
    "EMA50",

    "BB_High",
    "BB_Low",

    "ATR"
]

# ==========================================
# TARGET COLUMN
# ==========================================

TARGET_COLUMN = "Target_Return"

# ==========================================
# TRAIN / VALID / TEST SPLITS
# ==========================================

TRAIN_SPLIT = 0.70

VALID_SPLIT = 0.15

TEST_SPLIT = 0.15

# ==========================================
# MODEL CONFIG
# ==========================================

INPUT_DIM = len(FEATURE_COLUMNS)

D_MODEL = 64

NHEAD = 4

NUM_LAYERS = 2

DROPOUT = 0.2

# ==========================================
# TRAINING CONFIG
# ==========================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.0001

WEIGHT_DECAY = 1e-5

# ==========================================
# PATHS
# ==========================================

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "qqq_data.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "saved_models",
    "transformer_model.pth"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "scalers",
    "feature_scaler.pkl"
)

# ==========================================
# DEBUGGING PATHS
# ==========================================

print("\nDATA PATH:")
print(DATA_PATH)

print("\nMODEL PATH:")
print(MODEL_PATH)

print("\nSCALER PATH:")
print(SCALER_PATH)

print("\nMODEL EXISTS:")
print(os.path.exists(MODEL_PATH))

print("\nSCALER EXISTS:")
print(os.path.exists(SCALER_PATH))