# 📈 QQQ Transformer Forecasting Platform

An end-to-end AI-powered quantitative forecasting platform that predicts next-day market movement for the NASDAQ-100 ETF using Transformer Neural Networks, real-time financial data, and deep learning.

---

#  Live Demo

## 🌐 Streamlit Dashboard

https://qqq-transformer-forecasting-mjfjpzw3p2bk3nq8lfadg7.streamlit.app/

## ⚡ FastAPI Backend

https://qqq-transformer-forecasting.onrender.com

## 📚 API Documentation

https://qqq-transformer-forecasting.onrender.com/docs

---

# 📌 Project Overview

This project is a production-style AI forecasting system designed for quantitative finance applications.

The platform:

- Fetches live market data from Yahoo Finance
- Performs automated feature engineering
- Uses Transformer Neural Networks for time-series forecasting
- Predicts next-day returns for QQQ
- Generates AI-based BUY / HOLD / SELL signals
- Deploys a live FastAPI backend on Render
- Provides a professional Streamlit dashboard frontend

The system is inspired by modern institutional quantitative research workflows.

---

# 🧠 Key Features

## ✅ AI & Machine Learning

- Transformer Encoder architecture
- Multi-Head Self Attention
- Positional Encoding
- Deep Learning-based time-series forecasting
- Financial feature engineering
- Sequence modeling
- Directional prediction analysis

## ✅ Quantitative Finance Features

- Technical indicators
- Volatility analysis
- Momentum analysis
- Rolling statistics
- AI trading signals
- Confidence estimation
- Trend analysis

## ✅ Production Engineering

- FastAPI backend
- Streamlit frontend
- Docker deployment
- Render cloud deployment
- REST API architecture
- Real-time inference
- Modular ML pipeline

---

# 🏗️ System Architecture

```text
Users
   ↓
Streamlit Quant Dashboard
   ↓
FastAPI Backend
   ↓
Transformer Forecast Engine
   ↓
Feature Engineering Pipeline
   ↓
Yahoo Finance Live Market Data
```

---

# 📊 Machine Learning Pipeline

## 1. Data Collection

Market data is fetched using Yahoo Finance:

- Open
- High
- Low
- Close
- Volume

Ticker used:

```text
QQQ (NASDAQ-100 ETF)
```

---

## 2. Feature Engineering

The system automatically generates quantitative features.

### Technical Indicators

- RSI
- MACD
- EMA20
- EMA50
- Bollinger Bands
- ATR

### Statistical Features

- Returns
- Log Returns
- Rolling Mean
- Rolling Standard Deviation
- Rolling Volatility

---

## 3. Transformer Forecasting Model

The project uses a Transformer Encoder architecture implemented in PyTorch.

### Model Components

- Input Projection Layer
- Positional Encoding
- Multi-Head Attention
- Transformer Encoder Layers
- Regression Head

### Why Transformers?

Transformers are powerful for financial time-series forecasting because they:

- capture long-range dependencies
- learn temporal relationships
- model volatility regimes
- process sequences efficiently
- outperform traditional sequential models in many forecasting tasks

---

# 🧮 Self Attention Formula

The Transformer uses scaled dot-product attention:

```math
Attention(Q,K,V)=softmax(QK^T/√d_k)V
```

This enables the model to dynamically learn which historical periods are most relevant for forecasting.

---

# 📈 Dashboard Features

The Streamlit dashboard includes:

- Live QQQ market price
- Candlestick charts
- EMA trend analysis
- Volatility visualization
- AI prediction panel
- BUY / HOLD / SELL signals
- Confidence estimation
- Market statistics

The UI is styled with a professional quant-inspired dark theme.

---

# 🛠️ Tech Stack

## Backend

- Python
- FastAPI
- PyTorch
- Scikit-learn
- Pandas
- NumPy
- yfinance
- ta

## Frontend

- Streamlit
- Plotly

## Deployment

- Docker
- Render
- Streamlit Cloud
- GitHub

---

# 📂 Project Structure

```text
quant_transformer_forecasting/
│
├── backend/
│   ├── main.py
│   ├── predict.py
│   ├── preprocess.py
│   ├── model.py
│   ├── train.py
│   ├── config.py
│   └── requirements.txt
│
├── frontend/
│   ├── streamlit_app.py
│   ├── requirements.txt
│   └── .streamlit/
│       └── config.toml
│
├── saved_models/
│   └── transformer_model.pth
│
├── scalers/
│   └── feature_scaler.pkl
│
├── data/
│   └── qqq_data.csv
│
├── Dockerfile
│
└── README.md
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/qqq-transformer-forecasting.git

cd qqq-transformer-forecasting
```

---

## 2. Create Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / Mac

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Backend Dependencies

```bash
cd backend

pip install -r requirements.txt
```

---

## 4. Run Backend

```bash
uvicorn main:app --reload
```

Backend runs at:

```text
http://127.0.0.1:8000
```

---

## 5. Run Frontend

Open a second terminal:

```bash
cd frontend

streamlit run streamlit_app.py
```

---

# 🌐 API Endpoints

## Root Endpoint

```http
GET /
```

Returns API status.

---

## Health Check

```http
GET /health
```

Returns service health.

---

## Prediction Endpoint

```http
POST /predict
```

### Example Response

```json
{
  "status": "success",
  "data": {
    "ticker": "QQQ",
    "current_price": 711.23,
    "predicted_return": 0.000566,
    "predicted_price": 711.63,
    "signal": "HOLD",
    "confidence": 0.06
  }
}
```

---

# 📉 Model Performance

Example metrics achieved during training:

| Metric | Value |
|---|---|
| MAE | 0.009566 |
| RMSE | 0.013366 |
| Directional Accuracy | 55.93% |

In financial forecasting, directional accuracy above 55% can already be meaningful due to the highly noisy nature of markets.

---

# 🚀 Deployment

## Backend Deployment

Deployed using:

- Docker
- Render
- Gunicorn
- FastAPI

## Frontend Deployment

Deployed using:

- Streamlit Cloud

---

# 🔮 Future Improvements

Potential future enhancements include:

- Multi-stock forecasting
- Attention heatmap visualization
- Portfolio optimization
- Backtesting engine
- Sharpe ratio analysis
- Sentiment analysis integration
- Reinforcement learning trading agents
- Regime detection systems
- Live websocket streaming
- Kubernetes deployment
- CI/CD automation

---

# 🎯 Use Cases

This project is suitable for:

- Quantitative Finance
- AI Engineering
- Machine Learning Engineering
- MLOps Engineering
- Time-Series Forecasting
- Financial Data Science
- Portfolio Projects
- Research Demonstrations

---

# 👨‍💻 Author

Harshith Devraj

## Skills Demonstrated

- Deep Learning
- Transformers
- Quantitative Finance
- Time-Series Forecasting
- FastAPI
- Streamlit
- Docker
- Cloud Deployment
- MLOps
- Production AI Systems

---

# 📜 License

This project is intended for educational and research purposes.

---

# ⭐ Acknowledgements

Libraries and tools used in this project:

- PyTorch
- FastAPI
- Streamlit
- Yahoo Finance
- Plotly
- Scikit-learn
- Render
- Streamlit Cloud

---

# ⚠️ Disclaimer

This project is for educational and research purposes only.

It does not constitute financial advice or investment recommendations.

Trading financial markets involves risk.
