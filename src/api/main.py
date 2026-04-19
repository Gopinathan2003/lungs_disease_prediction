import io
import json
import time
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Histogram, start_http_server
from torchvision import transforms

DEFAULT_LABELS = [
    "BACTERIAL_PNEUMONIA",
    "CORONA_VIRUS_DISEASE",
    "NORMAL",
    "TUBERCULOSIS",
    "VIRAL_PNEUMONIA",
]
BASELINE_STATS_PATH = Path("data/processed/baseline_stats.json")

app = FastAPI(title="Lung Disease Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus
PREDICTIONS = Counter("predictions_total", "Total predictions")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
REQUESTS = Counter("http_requests_total", "Total HTTP requests")


def load_labels():
    if BASELINE_STATS_PATH.exists():
        with BASELINE_STATS_PATH.open() as f:
            baseline = json.load(f)
        labels = baseline.get("labels")
        if labels:
            return labels
    return DEFAULT_LABELS


LABELS = load_labels()
PREPROCESS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load latest MLflow model (change RUN_ID after training if needed)
model = mlflow.pyfunc.load_model("models:/lung_disease_model/latest")


def decode_prediction(raw_prediction):
    values = np.asarray(raw_prediction)
    if values.ndim == 0:
        class_idx = int(values.item())
        confidence = 1.0
    else:
        flattened = values.squeeze()
        if flattened.ndim == 0:
            class_idx = int(flattened.item())
            confidence = 1.0
        elif flattened.shape[0] == len(LABELS):
            class_idx = int(np.argmax(flattened))
            confidence = float(flattened[class_idx])
        else:
            class_idx = int(flattened[0])
            confidence = 1.0

    if class_idx < 0 or class_idx >= len(LABELS):
        raise HTTPException(500, f"Predicted class index {class_idx} is out of range for {len(LABELS)} labels")

    return LABELS[class_idx], confidence


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    return {"status": "ready", "labels": LABELS}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    REQUESTS.inc()

    if not file.content_type or not file.content_type.startswith("image"):
        raise HTTPException(400, "Image file required")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(400, "Invalid image file") from exc

    tensor = PREPROCESS(image).unsqueeze(0)
    raw_prediction = model.predict(tensor.numpy())
    prediction, confidence = decode_prediction(raw_prediction)

    latency = time.time() - start
    LATENCY.observe(latency)
    PREDICTIONS.inc()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "labels": LABELS,
        "latency_ms": round(latency * 1000, 2),
    }


if __name__ == "__main__":
    start_http_server(8001)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
