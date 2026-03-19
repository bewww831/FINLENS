import warnings, matplotlib
warnings.filterwarnings('ignore')
matplotlib.use("Agg")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from xgboost_model import predict_xgboost
from sentiment_model import predict_sentiment
from cnn_model import predict_cnn
from decision import run_decision_engine

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory="../assets"), name="static")

class TickerRequest(BaseModel):
    ticker: str

@app.get("/")
def serve_frontend():
    return FileResponse("../index.html")

@app.get("/health")
def health():
    return {"status": "ok", "models": ["xgboost", "finbert", "cnn"]}

@app.post("/predict")
def predict(req: TickerRequest):
    ticker = req.ticker.upper().strip()
    try:
        ts     = predict_xgboost(ticker)
        sent   = predict_sentiment(ticker)
        cnn    = predict_cnn(ticker)
        engine = run_decision_engine(ts, sent, cnn)
        return JSONResponse(content={
            "ticker":     ticker,
            "status":     "success",
            "timeseries": ts,
            "sentiment":  sent,
            "cnn":        cnn,
            "decision":   engine
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})