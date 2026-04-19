from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlflow.pyfunc
import torch
from PIL import Image
import io
from prometheus_client import start_http_server, Counter, Histogram
import time

app = FastAPI(title="Lung Disease Prediction API")

# Prometheus
PREDICTIONS = Counter("predictions_total", "Total predictions")
LATENCY = Histogram("prediction_latency_seconds", "Prediction latency")
REQUESTS = Counter("http_requests_total", "Total HTTP requests")

# Load latest MLflow model (change RUN_ID after training)
model = mlflow.pyfunc.load_model("models:/lung_disease_model/latest")  # or runs:/<run_id>/model

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/ready")
async def ready():
    return {"status": "ready"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    REQUESTS.inc()
    if not file.content_type.startswith("image"):
        raise HTTPException(400, "Image file required")
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    # Preprocess
    transform = torch.nn.Sequential(
        torch.nn.Resize((224,224)),
        torch.nn.ToTensor(),
        torch.nn.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    )
    tensor = transform(image).unsqueeze(0)
    # Predict via MLflow pyfunc
    pred = model.predict(tensor.numpy())
    prediction = "PNEUMONIA" if pred[0] == 1 else "NORMAL"
    confidence = float(max(pred)) if hasattr(pred, "__len__") else 0.95

    LATENCY.observe(time.time() - start)
    PREDICTIONS.inc()
    return {"prediction": prediction, "confidence": round(confidence, 4), "latency_ms": round((time.time()-start)*1000, 2)}

# Start Prometheus metrics server on port 8001
if __name__ == "__main__":
    start_http_server(8001)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)