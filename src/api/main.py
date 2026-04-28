import io
import json
import os
import time
from pathlib import Path

import mlflow.pyfunc
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from torchvision import transforms

DEFAULT_LABELS = [
    "BACTERIAL_PNEUMONIA",
    "CORONA_VIRUS_DISEASE",
    "NORMAL",
    "TUBERCULOSIS",
    "VIRAL_PNEUMONIA",
]
BASELINE_STATS_PATH = Path("data/processed/baseline_stats.json")
MODEL_URI = os.getenv("MODEL_URI", "models/latest")
APP_START_TIME = time.time()

app = FastAPI(title="Lung Disease Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus
REQUESTS = Counter("lung_http_requests_total", "Total HTTP requests", ["endpoint", "method", "status"])
PREDICTIONS = Counter("lung_predictions_total", "Total successful predictions")
PREDICTION_FAILURES = Counter("lung_prediction_failures_total", "Total failed prediction attempts", ["reason"])
LATENCY = Histogram("lung_prediction_latency_seconds", "Prediction latency")
MODEL_READY = Gauge("lung_model_ready", "Model readiness gauge")
LAST_SUCCESSFUL_PREDICTION = Gauge("lung_last_successful_prediction_timestamp", "Unix timestamp of last successful prediction")
APP_UPTIME = Gauge("lung_app_uptime_seconds", "API uptime in seconds")
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


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

# Prefer a local packaged model when running in Docker; MODEL_URI can still be
# overridden for registry-backed deployments.
model = None
try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    MODEL_READY.set(1)
except Exception:
    MODEL_READY.set(0)


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
    APP_UPTIME.set(time.time() - APP_START_TIME)
    REQUESTS.labels(endpoint="/health", method="GET", status="200").inc()
    return {"status": "healthy"}


@app.get("/ready")
async def ready():
    APP_UPTIME.set(time.time() - APP_START_TIME)
    if model is None:
        REQUESTS.labels(endpoint="/ready", method="GET", status="503").inc()
        raise HTTPException(503, "Model is not ready")
    REQUESTS.labels(endpoint="/ready", method="GET", status="200").inc()
    return {"status": "ready", "labels": LABELS, "model_uri": MODEL_URI}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    APP_UPTIME.set(time.time() - APP_START_TIME)

    if model is None:
        PREDICTION_FAILURES.labels(reason="model_not_ready").inc()
        REQUESTS.labels(endpoint="/predict", method="POST", status="503").inc()
        raise HTTPException(503, "Model is not loaded")

    if not file.content_type or not file.content_type.startswith("image"):
        PREDICTION_FAILURES.labels(reason="invalid_content_type").inc()
        REQUESTS.labels(endpoint="/predict", method="POST", status="400").inc()
        raise HTTPException(400, "Image file required")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError as exc:
        PREDICTION_FAILURES.labels(reason="invalid_image").inc()
        REQUESTS.labels(endpoint="/predict", method="POST", status="400").inc()
        raise HTTPException(400, "Invalid image file") from exc

    try:
        tensor = PREPROCESS(image).unsqueeze(0)
        raw_prediction = model.predict(tensor.numpy())
        prediction, confidence = decode_prediction(raw_prediction)
    except Exception as exc:
        PREDICTION_FAILURES.labels(reason="inference_error").inc()
        REQUESTS.labels(endpoint="/predict", method="POST", status="500").inc()
        raise HTTPException(500, f"Inference failed: {exc}") from exc

    latency = time.time() - start
    LATENCY.observe(latency)
    PREDICTIONS.inc()
    LAST_SUCCESSFUL_PREDICTION.set(time.time())
    REQUESTS.labels(endpoint="/predict", method="POST", status="200").inc()

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4),
        "labels": LABELS,
        "latency_ms": round(latency * 1000, 2),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=False)
