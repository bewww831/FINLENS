# FinLens: AI-Powered Financial Advisor Bot

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-FF6600?logo=python)](https://xgboost.readthedocs.io/)
[![FinBERT](https://img.shields.io/badge/FinBERT-Fine--tuned-FFD21E?logo=huggingface)](https://huggingface.co/project-aps/finbert-finetune)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)

_UoL BSc Computer Science CM3070 Final Project — Multi-modal AI investment recommendation system combining time-series forecasting, financial news sentiment analysis, and candlestick chart pattern recognition._

## 📖 Project Overview

Retail traders and individual investors lack access to affordable, transparent, and AI-powered financial advisory tools. Professional platforms like Bloomberg Terminal are prohibitively expensive, while free alternatives are unreliable, opaque, or narrowly focused on a single data source.

**FinLens** fills this gap:

- Combines **three independent AI models** operating on different data types into a unified recommendation.
- Implements a **transparent late fusion decision engine** every recommendation shows exactly how each model contributed.
- Accessible through a **standard web browser** no special hardware or software required.

⚡ **Impact:** Turns multi-signal financial analysis that was previously only available to institutional investors into a free, explainable, web-based tool for everyday traders.

## ✨ Key Features

- **🤖 Multi-Modal AI Fusion**
  - XGBoost: time-series price forecasting over a 21-day horizon
  - Fine-tuned FinBERT: financial news sentiment classification
  - ResNet-18 CNN: candlestick chart pattern recognition

- **🔢 Transparent Decision Engine**
  Late fusion weighted formula — every score is visible to the user:
```
  Final Score = 0.525 × score_ts + 0.325 × score_sent + 0.150 × score_cnn
```

- **📰 Live News Sentiment**
  5 latest news headlines per ticker, each classified as Positive / Negative / Neutral with confidence score.

- **📊 Seven-Level Recommendation Spectrum**
  Strong Buy → Buy → Hold → Sell → Strong Sell with plain-language explanations.

- **🌐 Web-Based & Free**
  No paywall, no installation — runs in any browser via FastAPI + plain HTML/CSS/JS frontend.

## 🏗️ Architecture

**Stack:**
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** FastAPI + Uvicorn
- **AI Models:** XGBoost, Fine-tuned FinBERT, ResNet-18 (PyTorch)
- **Data Source:** yfinance (historical prices, SPY/VIX context, news headlines)

### 🤖 AI Workflow
```
User inputs ticker
        ↓
┌───────────────────────────────────────┐
│  XGBoost      → Buy / Hold / Sell     │  weight: 52.5%
│  FinBERT      → Pos / Neu / Neg       │  weight: 32.5%
│  ResNet-18    → Bullish / Neutral / Bearish │  weight: 15.0%
└───────────────────────────────────────┘
        ↓
  Late Fusion Decision Engine
        ↓
  Final Score [0–1] → Recommendation + Explanation
```

Each model's output is normalised using:
```
score = ((P_buy − P_sell) × (1 − P_hold) + 1) / 2
```

## 📂 Repo Structure
```
FINLENS/
├── index.html              
├── assets/
│   ├── css/
│   │   └── styles.css
│   └── js/
│       └── app.js
└── backend/
    ├── app.py              # FastAPI routes
    ├── xgboost_model.py    # Time-series prediction
    ├── sentiment_model.py  # FinBERT sentiment
    ├── cnn_model.py        # ResNet-18 chart pattern
    ├── decision.py         # Late fusion engine
    ├── features.py         # Feature engineering (33 indicators)
    ├── test.py             # API tests
    └── trained/
        ├── xgboost.pkl
        └── cnn.pt
```

## 🚀 Installation

### Prerequisites
- Python 3.11+

### Setup
```bash
# Clone the repo
git clone https://github.com/yourusername/finlens.git
cd finlens

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r backend/requirements.txt
```

### Running
```bash
cd backend
uvicorn app:app --reload
```

Then open `http://127.0.0.1:8000` in your browser.

> 💡 **First run tip:** The first request may take 30–60 seconds as FinBERT downloads its weights from Hugging Face. Subsequent requests will be faster as models are cached in memory.

## 🧪 Testing

Make sure uvicorn is running first, then in a second terminal:
```bash
cd backend
python test.py
```

Expected output:
```
Running tests... (ensure uvicorn is running)

[PASS] Health check
[PASS] Predict AAPL recommendation: HOLD
[PASS] Invalid ticker handled gracefully
[PASS] Decision scores valid final: 0.44969
[PASS] Sentiment articles valid: 5 articles returned
```

## 📊 Evaluation Results

| Model | Metric | Result |
|---|---|---|
| **XGBoost (Tuned)** | Holdout Accuracy | 95% |
| **XGBoost (Tuned)** | Backtesting Portfolio Return | 141.17% |
| **XGBoost (Tuned)** | Tickers beating buy-and-hold | 9 / 16 |
| **Fine-tuned FinBERT** | Test Accuracy | 89.38% |
| **Fine-tuned FinBERT** | Macro F1-Score | 87.44% |
| **ResNet-18 CNN** | Test Accuracy | 71.69% |
| **ResNet-18 CNN** | Directional Recall (Bullish/Bearish) | 84% |

Backtesting conducted over December 2022 – December 2025 across 16 tickers (AAPL, MSFT, GOOGL, AMZN, META, TSLA, NVDA, JPM, BAC, GS, XOM, CVX, JNJ, PFE, WMT, HD) with $160,000 total invested capital.

> ⚠️ Past performance does not guarantee future results. This tool is for educational purposes only and does not constitute financial advice.

## 🔮 Future Work

- Aggregate multiple headlines per ticker for a more robust sentiment score
- Expand XGBoost training beyond the current 16 tickers
- Retrain CNN on a more diverse multi-asset candlestick dataset
- Add historical prediction log to track past recommendations vs actual price movements
- Push notifications and mobile support
