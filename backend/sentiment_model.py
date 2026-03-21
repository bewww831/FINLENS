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
    device="mps" if torch.backends.mps.is_available() else (0 if torch.cuda.is_available() else -1),
    truncation=True,
    max_length=512,
    top_k=None
)

def predict_sentiment(ticker):
    news        = yf.Ticker(ticker).news[:5]
    articles    = []
    first_label = None

    for i, item in enumerate(news):
        headline = item["content"]["title"]
        source   = item["content"].get("provider", {}).get("displayName", "Unknown")
        summary  = item["content"].get("summary", "")
        scores   = {x["label"].lower(): round(x["score"], 5) for x in finbert_pipe(headline)[0]}
        label    = max(scores, key=scores.get)
        conf     = round(scores[label] * 100, 2)

        if i == 0:
            first_label = label

        url = item["content"].get("canonicalUrl", {}).get("url", "") or \
              item["content"].get("clickThroughUrl", {}).get("url", "")

        articles.append({
            "headline": headline,
            "source":   source,
            "summary":  summary,
            "url":      url,
            "label":    label,
            "conf":     conf,
            "positive": scores.get("positive", 0),
            "negative": scores.get("negative", 0),
            "neutral":  scores.get("neutral",  0),
        })

    # first headline drives the score used in decision engine
    first = articles[0]
    return {
        "headline": first["headline"],
        "positive": first["positive"],
        "negative": first["negative"],
        "neutral":  first["neutral"],
        "label":    first_label,
        "articles": articles
    }