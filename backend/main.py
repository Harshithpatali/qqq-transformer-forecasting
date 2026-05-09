from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from predict import predict_next_day

# ==========================================
# FASTAPI APP
# ==========================================

app = FastAPI(

    title="QQQ Transformer Forecast API",

    description=(
        "AI-powered NASDAQ forecasting "
        "using Transformer Neural Networks"
    ),

    version="1.0.0"
)

# ==========================================
# ENABLE CORS
# ==========================================

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)

# ==========================================
# ROOT ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {

        "message":
        "QQQ Transformer Forecast API Running",

        "status":
        "success"
    }


# ==========================================
# HEALTH CHECK
# ==========================================

@app.get("/health")
def health():

    return {

        "status": "healthy"
    }


# ==========================================
# PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict():

    try:

        prediction = predict_next_day()

        return {

            "status": "success",

            "data": prediction
        }

    except Exception as e:

        return {

            "status": "error",

            "message": str(e)
        }