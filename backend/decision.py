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
    ts_score   = normalise_score(ts["buy"],        ts["sell"],      ts["hold"])
    sent_score = normalise_score(sent["positive"], sent["negative"], sent["neutral"])
    cnn_score  = normalise_score(cnn["bullish"],   cnn["bearish"],  cnn["neutral"])

    w_ts, w_sent, w_cnn = 0.525, 0.325, 0.150
    ts_contrib   = round(w_ts   * ts_score,   5)
    sent_contrib = round(w_sent * sent_score, 5)
    cnn_contrib  = round(w_cnn  * cnn_score,  5)
    final_score  = round(ts_contrib + sent_contrib + cnn_contrib, 5)

    rec, emoji, explanation = get_recommendation(final_score)
    return {
        "ts_score":       ts_score,
        "sent_score":     sent_score,
        "cnn_score":      cnn_score,
        "ts_contrib":     ts_contrib,
        "sent_contrib":   sent_contrib,
        "cnn_contrib":    cnn_contrib,
        "final_score":    final_score,
        "recommendation": rec,
        "emoji":          emoji,
        "explanation":    explanation
    }