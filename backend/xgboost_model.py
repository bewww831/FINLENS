import os, numpy as np, joblib, yfinance as yf
from features import build_features

BASE_DIR  = os.path.dirname(__file__)
XGB_PATH  = os.path.join(BASE_DIR, "trained", "xgboost.pkl")

xgb_bundle = joblib.load(XGB_PATH)
xgb_model  = xgb_bundle["model"]
xgb_scaler = xgb_bundle["scaler"]
xgb_feats  = xgb_bundle["features"]

def predict_xgboost(ticker):
    raw = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    spy = yf.download("SPY",  period="2y", auto_adjust=True, progress=False)
    vix = yf.download("^VIX", period="2y", auto_adjust=True, progress=False)
    raw.columns = [c[0] if isinstance(c, tuple) else c for c in raw.columns]
    df       = build_features(raw, spy, vix).dropna(subset=xgb_feats)
    X_scaled = xgb_scaler.transform(df[xgb_feats].iloc[[-1]])
    proba    = xgb_model.predict_proba(X_scaled)[0]
    return {
        "sell":  round(float(proba[0]), 5),
        "hold":  round(float(proba[1]), 5),
        "buy":   round(float(proba[2]), 5),
        "label": ["sell", "hold", "buy"][int(np.argmax(proba))]
    }