import torch, yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

label_map    = {0: "neutral", 1: "negative", 2: "positive"}
ft_tokenizer = AutoTokenizer.from_pretrained("project-aps/finbert-finetune")
ft_model     = AutoModelForSequenceClassification.from_pretrained("project-aps/finbert-finetune")
ft_model.config.id2label = label_map
ft_model.config.label2id = {v: k for k, v in label_map.items()}

finbert_pipe = pipeline(
    "text-classification",
    model=ft_model,
    tokenizer=ft_tokenizer,
    device=0 if torch.cuda.is_available() else -1,
    truncation=True,
    max_length=512,
    top_k=None
)

def predict_sentiment(ticker):
    headline = yf.Ticker(ticker).news[0]["content"]["title"]
    scores   = {item["label"].lower(): round(item["score"], 5) for item in finbert_pipe(headline)[0]}
    return {
        "headline": headline,
        "positive": scores.get("positive", 0),
        "negative": scores.get("negative", 0),
        "neutral":  scores.get("neutral",  0),
        "label":    max(scores, key=scores.get)
    }