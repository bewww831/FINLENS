import io, os, torch, torch.nn as nn, numpy as np
import yfinance as yf, mplfinance as mpf, matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image

BASE_DIR = os.path.dirname(__file__)
CNN_PATH = os.path.join(BASE_DIR, "trained", "cnn.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

cnn_model = resnet18(pretrained=False)
cnn_model.fc = nn.Linear(512, 3)
cnn_model.load_state_dict(torch.load(CNN_PATH, map_location=DEVICE, weights_only=True))
cnn_model = cnn_model.to(DEVICE).eval()

cnn_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def predict_cnn(ticker):
    df = yf.download(ticker, period="3mo", interval="1d", auto_adjust=True, progress=False)
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    df = df.tail(30)[["Open", "High", "Low", "Close", "Volume"]]
    fig, _ = mpf.plot(df, type="candle", style="charles", figsize=(2.24, 2.24),
                      axisoff=True, volume=False, returnfig=True)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    x = cnn_transform(Image.open(buf).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(cnn_model(x), dim=1).squeeze(0).cpu().tolist()
    return {
        "neutral": round(probs[0], 5),
        "bearish": round(probs[1], 5),
        "bullish": round(probs[2], 5),
        "label":   ["neutral", "bearish", "bullish"][int(np.argmax(probs))]
    }