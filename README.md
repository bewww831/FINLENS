<div align="center">
  <img src="assets/finlens.png" alt="FinLens Logo" width="200"/>
</div>

# FinLens: A Multi-Modal AI Financial Advisory

[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-FF6600?logo=python)](https://xgboost.readthedocs.io/)
[![FinBERT](https://img.shields.io/badge/FinBERT-Fine--tuned-FFD21E?logo=huggingface)](https://huggingface.co/project-aps/finbert-finetune)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)](#testing)

_UoL BSc Computer Science CM3070 Final Project._

## Project Overview

Retail traders and individual investors lack access to affordable, transparent, and AI-powered financial advisory tools. Professional platforms like Bloomberg Terminal are expensive, while free alternatives are unreliable, opaque, or narrowly focused on a single data source.

**FinLens** fills this gap:

- Combines **three independent AI models** operating on different data types into a unified recommendation.
- Implements a **transparent late fusion decision engine** every recommendation shows exactly how each model contributed.
- Accessible through a **standard web browser** no special hardware or software required.

By combining multiple models through a structured late fusion strategy, FinLens provides both predictive performance and interpretability.

## Key Features

- **Multi-Modal AI Fusion**
  - XGBoost: time-series price forecasting over a 21-day horizon
  - Fine-tuned FinBERT: financial news sentiment classification
  - ResNet-18 CNN: candlestick chart pattern recognition

- **Transparent Decision Engine**
  Model outputs are combined using a weighted late fusion approach:
```
  Final Score = 0.525 × score_ts + 0.325 × score_sent + 0.150 × score_cnn
```
Where each model score is normalized to a common scale:
```
score = ((P_buy − P_sell) × (1 − P_hold) + 1) / 2
```

- **Live News Sentiment**
  5 latest news headlines per ticker, each classified as Positive / Negative / Neutral with confidence score.

- **Seven-Level Recommendation Spectrum**
  Strong Buy → Buy → Hold → Sell → Strong Sell with plain-language explanations.

- **Web-Based & Free**
  No paywall, no installation, runs in any browser via FastAPI + plain HTML/CSS/JS frontend.

## Architecture

**Stack:**
- **Frontend:** HTML, CSS, JavaScript
- **Backend:** FastAPI + Uvicorn
- **AI Models:** XGBoost, Fine-tuned FinBERT, ResNet-18 (PyTorch)
- **Data Source:** yfinance (price data, market context, news headlines)

### System Architecture Diagram

![FinLens System Architecture](assets/architecture.png)

### AI Workflow
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

## Repository Structure
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
    ├── cnn_model.py        # CNN chart pattern
    ├── decision.py         # Late fusion engine
    ├── features.py         # Feature engineering 
    ├── test.py             # API tests
    ├── requirements.txt
    └── trained/
        ├── xgboost.pkl
        └── cnn.pt
```

## Installation

### Prerequisites
- Python 3.11+

### Setup
```bash
# Clone the repo
git clone https://github.com/bewww831/FINLENS.git
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
uvicorn app:app --reload --port 8000
```

Then open `http://127.0.0.1:8000` in your browser.

> **First run tip:** The first request may take 30–60 seconds as FinBERT downloads its weights from Hugging Face. Subsequent requests will be faster as models are cached in memory.

## Testing

Make sure uvicorn is running first, then in a second terminal:
```bash
cd backend
python test.py
```

Expected output:
```
Running tests... (ensure uvicorn is running)

── Health ──
[PASS] Health check
[PASS] Health response time: 0.002s
[PASS] Health lists all three models

── Predict — Valid ──
[PASS] Predict AAPL recommendation: SELL
[PASS] Predict response time: 0.8s
[PASS] Response structure complete — all keys present

── Predict — Edge Cases ──
[PASS] Invalid ticker handled gracefully
[PASS] Lowercase ticker normalised correctly
[PASS] Ticker with surrounding spaces handled correctly
[PASS] Empty ticker handled gracefully

── Probability Distributions ──
[PASS] XGBoost probabilities sum to ~1.0 (1.00000)
[PASS] FinBERT probabilities sum to ~1.0 (0.99999)
[PASS] CNN probabilities sum to ~1.0 (0.99999)

── Decision Engine ──
[PASS] Decision scores valid final: 0.28917
[PASS] All scores and contributions within valid ranges
[PASS] Decision explanation and emoji present

── Sentiment Articles ──
[PASS] Sentiment articles valid: 5 articles returned
[PASS] Sentiment articles all contain url field

── Prices Endpoint ──
[PASS] Prices endpoint returned 16 tickers
[PASS] Prices structure valid for all 16 tickers

── Multiple Tickers ──
[PASS] Predict MSFT: HOLD
[PASS] Predict TSLA: HOLD
[PASS] Predict NVDA: HOLD

21 passed, 0 failed
[PASS] All tests passed
```

## Evaluation Results

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

> **Disclaimer**:
This system is intended for educational and research purposes only. It does not constitute financial advice. Past performance is not indicative of future results.

## Future Work

- Expand dataset coverage across sectors and asset classes
- Incorporate multi-headline aggregation for sentiment robustness
- Improve CNN generalisation using larger, multi-market datasets
- Add historical prediction log to track past recommendations vs actual price movements
- Extend to mobile deployment and real-time notification systems
- Real-time price for placing orders
