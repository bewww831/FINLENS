# backend/app.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import yfinance as yf
import numpy as np
import pandas as pd
import mplfinance as mpf
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io, os, warnings
import matplotlib

warnings.filterwarnings('ignore')
matplotlib.use("Agg")
app = FastAPI()

# Allow frontend HTML to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── PATHS ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODEL_DIR  = os.path.join(BASE_DIR, "models")
XGB_PATH   = os.path.join(MODEL_DIR, "xgboost.pkl")
CNN_PATH   = os.path.join(MODEL_DIR, "cnn.pt")

# ── DEVICE ───────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# LOAD MODELS AT STARTUP
# ============================================================

# ── XGBoost ──────────────────────────────────────────────────
print("Loading XGBoost...")
xgb_bundle  = joblib.load(XGB_PATH)
xgb_model   = xgb_bundle["model"]
xgb_scaler  = xgb_bundle["scaler"]
xgb_feats   = xgb_bundle["features"]
print(f"✅ XGBoost loaded — {len(xgb_feats)} features")

# ── FinBERT ──────────────────────────────────────────────────
print("Loading FinBERT from Hugging Face...")
ft_tokenizer = AutoTokenizer.from_pretrained("project-aps/finbert-finetune")
ft_model     = AutoModelForSequenceClassification.from_pretrained("project-aps/finbert-finetune")
label_map    = {0: "neutral", 1: "negative", 2: "positive"}
ft_model.config.id2label  = label_map
ft_model.config.label2id  = {v: k for k, v in label_map.items()}
finbert_pipe = pipeline(
    "text-classification",
    model=ft_model,
    tokenizer=ft_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
    top_k=None
)
print("✅ FinBERT loaded")

# ── ResNet-18 CNN ─────────────────────────────────────────────
print("Loading CNN...")
cnn_model = resnet18(pretrained=False)
cnn_model.fc = nn.Linear(512, 3)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE))
cnn_model = cnn_model.to(DEVICE)
cnn_model.eval()
print("✅ CNN loaded")

# ── Image transform ───────────────────────────────────────────
cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# ============================================================
# FEATURE ENGINEERING
# ============================================================

FEATURE_COLS = [
    'ret_1d','ret_5d','ret_21d','ret_63d',
    'vol_10','vol_21','vol_63','vol_ratio',
    'price_vs_sma20','price_vs_sma50','price_vs_sma200',
    'sma20_vs_sma50','sma50_vs_sma200',
    'rsi14','rsi14_change',
    'macd_norm','macd_hist',
    'bb_pos','bb_width',
    'atr14_norm',
    'range_pct','gap_pct','close_pos',
    'vol_ratio_5d','vol_ratio_21d','obv_ratio',
    'spy_above_200ma','spy_ret_21d',
    'vix_level','vix_5d_change','vix_vs_sma20',
    'rs_vs_spy_21d','rs_vs_spy_5d',
]

def build_features(df, spy, vix):
    df = df.copy()
    df['ret_1d']  = df['Close'].pct_change(1)
    df['ret_5d']  = df['Close'].pct_change(5)
    df['ret_21d'] = df['Close'].pct_change(21)
    df['ret_63d'] = df['Close'].pct_change(63)
    df['vol_10']    = df['ret_1d'].rolling(10).std()
    df['vol_21']    = df['ret_1d'].rolling(21).std()
    df['vol_63']    = df['ret_1d'].rolling(63).std()
    df['vol_ratio'] = df['vol_10'] / df['vol_21'].replace(0, np.nan)
    sma20  = df['Close'].rolling(20).mean()
    sma50  = df['Close'].rolling(50).mean()
    sma200 = df['Close'].rolling(200).mean()
    df['price_vs_sma20']  = df['Close'] / sma20 - 1
    df['price_vs_sma50']  = df['Close'] / sma50 - 1
    df['price_vs_sma200'] = df['Close'] / sma200 - 1
    df['sma20_vs_sma50']  = sma20 / sma50 - 1
    df['sma50_vs_sma200'] = sma50 / sma200 - 1
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['rsi14']        = 100 - (100 / (1 + rs))
    df['rsi14_change'] = df['rsi14'].diff()
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    macd  = ema12 - ema26
    signal= macd.ewm(span=9).mean()
    df['macd_norm'] = macd / df['Close']
    df['macd_hist'] = (macd - signal) / df['Close']
    bb_mid = df['Close'].rolling(20).mean()
    bb_std = df['Close'].rolling(20).std()
    df['bb_pos']   = (df['Close'] - bb_mid) / (2 * bb_std).replace(0, np.nan)
    df['bb_width'] = (4 * bb_std) / bb_mid.replace(0, np.nan)
    hl  = df['High'] - df['Low']
    hpc = abs(df['High'] - df['Close'].shift())
    lpc = abs(df['Low']  - df['Close'].shift())
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df['atr14_norm'] = tr.rolling(14).mean() / df['Close']
    df['range_pct']  = (df['High'] - df['Low']) / df['Close'].shift()
    df['gap_pct']    = (df['Open'] - df['Close'].shift()) / df['Close'].shift()
    df['close_pos']  = (df['Close'] - df['Low']) / (df['High'] - df['Low']).replace(0, np.nan)
    vs5  = df['Volume'].rolling(5).mean()
    vs21 = df['Volume'].rolling(21).mean()
    df['vol_ratio_5d']  = df['Volume'] / vs5.replace(0, np.nan)
    df['vol_ratio_21d'] = df['Volume'] / vs21.replace(0, np.nan)
    df['obv_ratio']     = (df['Volume'] * np.sign(df['ret_1d'])).rolling(21).sum() / \
                           df['Volume'].rolling(21).sum().replace(0, np.nan)
    spy = spy.copy()
    spy.columns = [c[0] if isinstance(c, tuple) else c for c in spy.columns]
    spy_close  = spy['Close']
    spy_sma200 = spy_close.rolling(200).mean()
    spy_ret21  = spy_close.pct_change(21)
    df['spy_above_200ma'] = (spy_close > spy_sma200).astype(int).reindex(df.index).ffill()
    df['spy_ret_21d']     = spy_ret21.reindex(df.index).ffill()
    vix = vix.copy()
    vix.columns = [c[0] if isinstance(c, tuple) else c for c in vix.columns]
    vix_close = vix['Close']
    vix_sma20 = vix_close.rolling(20).mean()
    df['vix_level']     = vix_close.reindex(df.index).ffill()
    df['vix_5d_change'] = vix_close.pct_change(5).reindex(df.index).ffill()
    df['vix_vs_sma20']  = (vix_close / vix_sma20 - 1).reindex(df.index).ffill()
    df['rs_vs_spy_21d'] = df['ret_21d'] - spy_ret21.reindex(df.index).ffill()
    df['rs_vs_spy_5d']  = df['ret_5d']  - spy_close.pct_change(5).reindex(df.index).ffill()
    return df

# ============================================================
# PREDICTION FUNCTIONS
# ============================================================

def predict_xgboost(ticker):
    raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    spy = yf.download("SPY",  period="2y", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", period="2y", auto_adjust=True, progress=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    df = build_features(raw, spy, vix)
    df = df.dropna(subset=xgb_feats)
    X  = df[xgb_feats].iloc[[-1]]
    X_scaled = xgb_scaler.transform(X)
    proba    = xgb_model.predict_proba(X_scaled)[0]
    return {
        "sell": round(float(proba[0]), 5),
        "hold": round(float(proba[1]), 5),
        "buy":  round(float(proba[2]), 5),
        "label": ["sell", "hold", "buy"][int(np.argmax(proba))]
    }

def predict_sentiment(ticker):
    stock    = yf.Ticker(ticker)
    headline = stock.news[0]["content"]["title"]
    outputs  = finbert_pipe(headline)[0]
    scores   = {item["label"].lower(): round(item["score"], 5) for item in outputs}
    label    = max(scores, key=scores.get)
    return {
        "headline": headline,
        "positive": scores.get("positive", 0),
        "negative": scores.get("negative", 0),
        "neutral":  scores.get("neutral",  0),
        "label":    label
    }

def predict_cnn(ticker):
    df = yf.download(ticker, period="3mo", interval="1d",
                     auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.tail(30)[["Open","High","Low","Close","Volume"]]
    fig, _ = mpf.plot(df, type="candle", style="charles",
                      figsize=(2.24, 2.24), axisoff=True,
                      volume=False, returnfig=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img   = Image.open(buf).convert("RGB")
    x     = cnn_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(x), dim=1).squeeze(0).cpu().tolist()
    idx_to_name = ["neutral", "bearish", "bullish"]
    return {
        "neutral":  round(probs[0], 5),
        "bearish":  round(probs[1], 5),
        "bullish":  round(probs[2], 5),
        "label":    idx_to_name[int(np.argmax(probs))]
    }

# ============================================================
# DECISION ENGINE
# ============================================================

def normalise_score(buy, sell, hold):
    raw      = buy - sell
    dampened = raw * (1 - hold)
    return round((dampened + 1) / 2, 5)

def get_recommendation(score):
    if score >= 0.75:
        return "STRONG BUY",  "🟢", "Strong bullish signal. All models strongly agree the price will rise significantly."
    elif score >= 0.60:
        return "BUY",         "🟢", "Bullish signal. Positive signals outweigh negative ones across all models."
    elif score >= 0.53:
        return "HOLD",        "🟡", "Mild bullish signal. Some positive signs but not fully convincing."
    elif score >= 0.47:
        return "HOLD",        "🟡", "Neutral signal. Models are uncertain — mixed signals, best to wait and monitor."
    elif score >= 0.40:
        return "HOLD",        "🟡", "Mild bearish signal. Some negative signs but not fully convincing."
    elif score >= 0.25:
        return "SELL",        "🔴", "Bearish signal. Negative signals outweigh positive ones across all models."
    else:
        return "STRONG SELL", "🔴", "Strong bearish signal. All models strongly agree the price will drop significantly."

def run_decision_engine(ts, sent, cnn):
    ts_score   = normalise_score(ts["buy"],       ts["sell"],   ts["hold"])
    sent_score = normalise_score(sent["positive"], sent["negative"], sent["neutral"])
    cnn_score  = normalise_score(cnn["bullish"],   cnn["bearish"],  cnn["neutral"])
    w_ts, w_sent, w_cnn = 0.525, 0.325, 0.150
    ts_contrib   = round(w_ts   * ts_score,   5)
    sent_contrib = round(w_sent * sent_score, 5)
    cnn_contrib  = round(w_cnn  * cnn_score,  5)
    final_score  = round(ts_contrib + sent_contrib + cnn_contrib, 5)
    rec, emoji, explanation = get_recommendation(final_score)
    return {
        "ts_score":    ts_score,
        "sent_score":  sent_score,
        "cnn_score":   cnn_score,
        "ts_contrib":  ts_contrib,
        "sent_contrib":sent_contrib,
        "cnn_contrib": cnn_contrib,
        "final_score": final_score,
        "recommendation": rec,
        "emoji":       emoji,
        "explanation": explanation
    }

# ============================================================
# API ENDPOINTS
# ============================================================

class TickerRequest(BaseModel):
    ticker: str

@app.get("/")
def root():
    return {"status": "Financial Advisor Bot API is running"}

@app.get("/health")
def health():
    return {"status": "ok", "models": ["xgboost", "finbert", "cnn"]}

@app.post("/predict")
def predict(req: TickerRequest):
    ticker = req.ticker.upper().strip()
    try:
        print(f"\n🔄 Running prediction for {ticker}...")
        ts   = predict_xgboost(ticker)
        sent = predict_sentiment(ticker)
        cnn  = predict_cnn(ticker)
        engine = run_decision_engine(ts, sent, cnn)
        return JSONResponse(content={
            "ticker":    ticker,
            "status":    "success",
            "timeseries":  ts,
            "sentiment":   sent,
            "cnn":         cnn,
            "decision":    engine
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})