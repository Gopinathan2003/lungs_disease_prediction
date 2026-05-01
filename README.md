# Lung Disease Prediction from Chest X-Rays

An end-to-end MLOps project that predicts lung disease classes from chest X-ray images. The project includes model training, experiment tracking, API serving, a small web UI, orchestration with Airflow, and monitoring with Prometheus and Grafana.

This project is for learning and demonstration. It is not a medical diagnosis tool.

![Project pipeline](images/pipeline.png)

## What This Project Does

The application accepts a chest X-ray image and predicts one of these classes:

| Class |
|-------|
| Normal |
| Bacterial Pneumonia |
| Viral Pneumonia |
| Tuberculosis |
| Corona Virus Disease |

The full workflow is:

1. Prepare raw chest X-ray images.
2. Build processed datasets and baseline statistics.
3. Train a ResNet18 transfer-learning model.
4. Track experiments and model versions in MLflow.
5. Serve predictions through FastAPI.
6. Use a browser UI to upload images and see predictions.
7. Monitor API/model health with Prometheus and Grafana.
8. Run the training pipeline through Airflow.

## Screenshots

### Web Application

Use this page to upload an X-ray and call the FastAPI prediction endpoint.

![Web application](images/ui.png)

### MLflow Experiment Tracking

MLflow stores training parameters, metrics, artifacts, and registered model versions.

![MLflow dashboard](images/mlflow.png)

### Airflow Pipeline

Airflow orchestrates the data ingestion, feature building, and model training DAG.

![Airflow DAG](images/airflow.png)

### Prometheus Metrics

Prometheus scrapes the API metrics endpoint and stores operational metrics.

![Prometheus targets](images/prometheus.png)

### Grafana Dashboard

Grafana visualizes API availability, request rate, prediction latency, failures, and model readiness.

![Grafana dashboard](images/grafana.png)

## Technology Stack

| Area | Tool |
|------|------|
| Language | Python |
| Deep Learning | PyTorch, TorchVision |
| Model | ResNet18 transfer learning |
| API | FastAPI |
| Frontend | HTML, Bootstrap, JavaScript |
| Experiment Tracking | MLflow |
| Pipeline Orchestration | Apache Airflow |
| Data Pipeline | DVC |
| Monitoring | Prometheus, Grafana, Alertmanager |
| Containerization | Docker, Docker Compose |

## Project Structure

```text
lungs_disease_prediction/
├── airflow/
│   └── dags/
│       └── ml_pipeline.py
├── alertmanager/
│   └── alertmanager.yml
├── data/
│   ├── raw/
│   └── processed/
├── frontend/
│   └── index.html
├── grafana/
│   └── provisioning/
├── images/
│   ├── airflow.png
│   ├── grafana.png
│   ├── mlflow.png
│   ├── pipeline.png
│   ├── prometheus.png
│   └── ui.png
├── mlruns/
├── models/
│   └── latest/
├── prometheus/
│   ├── alerts.yml
│   ├── prometheus.yml
│   └── prometheus.docker.yml
├── src/
│   ├── api/
│   │   └── main.py
│   ├── data/
│   │   └── ingestion.py
│   ├── features/
│   │   └── build_features.py
│   └── models/
│       └── train.py
├── docker-compose.yml
├── dockerfile
├── dockerfile.airflow
├── requirements.txt
├── requirements-serving.txt
├── requirements-airflow.txt
└── README.md
```

## Prerequisites

Install these before running the project:

| Requirement | Why |
|-------------|-----|
| Python 3.10 to 3.13 | Run training, API, and local tools |
| Docker | Run services in containers |
| docker-compose | Start all services together |
| Git | Clone and version the project |
| DVC | Reproduce data pipeline, if using DVC locally |

Check Docker and Compose:

```bash
docker --version
docker-compose --version
```

This project is configured for the standalone command:

```bash
docker-compose up --build
```

If your machine has the newer Compose plugin, `docker compose up --build` may also work.

## Dataset Setup

Put the chest X-ray dataset here:

```text
data/raw/chest_xray/
```

Expected folder shape:

```text
data/raw/chest_xray/
├── train/
├── test/
└── val/
```

Each split should contain class folders with images inside them. The training script reads these folders using `torchvision.datasets.ImageFolder`.

## Quick Start with Docker

Use Docker if you want the easiest way to run all services.

### 1. Clone the Project

```bash
git clone <your-repo-url>
cd lungs_disease_prediction
```

### 2. Create the Environment File

```bash
cp .env.example .env
sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env
```

The `AIRFLOW_UID` value helps Airflow write logs and files correctly on Linux bind mounts.

Mailtrap values in `.env` are only needed if you want email alerts from Alertmanager.

### 3. Train or Provide a Model

The API expects a local model at:

```text
models/latest/
```

If the folder does not exist yet, train the model:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/models/train.py
```

Training creates:

```text
models/latest/
mlruns/
```

### 4. Stop Manual Services on the Same Ports

If you previously started services manually, stop them before running Docker Compose. These ports are used:

| Service | Port |
|---------|------|
| Airflow | 8080 |
| API | 8000 |
| Frontend | 8001 |
| MLflow UI | 5000 |
| MLflow model server | 5001 |
| Prometheus | 9090 |
| Grafana | 3001 |
| Alertmanager | 9093 |

### 5. Start Everything

```bash
sudo docker-compose up --build
```

If you hit the Docker Compose v1 error `KeyError: 'ContainerConfig'`, run:

```bash
sudo ./scripts/compose-up-clean.sh
```

That script runs `docker-compose down --remove-orphans` and then recreates the stack.

## Service URLs

| Service | URL | What to Use It For |
|---------|-----|--------------------|
| Web UI | http://localhost:8001 | Upload X-ray images |
| API docs | http://localhost:8000/docs | Test API endpoints |
| API metrics | http://localhost:8000/metrics | Raw Prometheus metrics |
| MLflow UI | http://localhost:5000 | View experiments and models |
| MLflow model server | http://localhost:5001 | Direct model serving endpoint |
| Airflow | http://localhost:8080 | Run or inspect the DAG |
| Prometheus | http://localhost:9090 | Inspect scraped metrics |
| Grafana | http://localhost:3001 | View monitoring dashboards |
| Alertmanager | http://localhost:9093 | View configured alerts |

## Login Notes

### Grafana

Default Docker login:

```text
username: admin
password: admin
```

### Airflow

Airflow standalone prints the generated login details in its container logs. To inspect them:

```bash
sudo docker-compose logs airflow --tail=200
```

## How to Test the Running App

1. Open the web UI: http://localhost:8001
2. Upload a chest X-ray image.
3. Click `Predict`.
4. Confirm that the page returns a class, confidence score, and latency.
5. Open API docs: http://localhost:8000/docs
6. Check API readiness: http://localhost:8000/ready
7. Check metrics: http://localhost:8000/metrics
8. Open Grafana: http://localhost:3001
9. Open MLflow: http://localhost:5000
10. Open Airflow: http://localhost:8080

## Run Without Docker

Use this path if you want to learn each component separately.

### 1. Create Python Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Prepare Data and Train

```bash
dvc repro
python src/models/train.py
```

If you do not use DVC, make sure your dataset is already available under `data/raw/chest_xray/`, then run:

```bash
python src/data/ingestion.py
python src/features/build_features.py
python src/models/train.py
```

### 3. Start Services in Separate Terminals

Terminal 1:

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Terminal 2:

```bash
mlflow models serve -m models:/lung_disease_model/latest -p 5001 --host 0.0.0.0 --no-conda
```

Terminal 3:

```bash
python3 src/api/main.py
```

Terminal 4:

```bash
prometheus --config.file=prometheus/prometheus.yml
```

Terminal 5:

```bash
airflow standalone
```

Terminal 6:

```bash
cd frontend
python3 -m http.server 8001
```

## Airflow DAG

The DAG file is:

```text
airflow/dags/ml_pipeline.py
```

It runs three tasks:

| Task | Script |
|------|--------|
| `ingest_data` | `python -m src.data.ingestion` |
| `build_features` | `python -m src.features.build_features` |
| `train_model` | `python -m src.models.train` |

In the Airflow UI, look for:

```text
lung_disease_training_pipeline
```

If the DAG is not visible, check import errors:

```bash
sudo docker-compose logs airflow --tail=200
```

## Monitoring

The API exposes Prometheus metrics at:

```text
http://localhost:8000/metrics
```

Important metrics include:

| Metric | Meaning |
|--------|---------|
| `lung_http_requests_total` | API request count |
| `lung_predictions_total` | Successful predictions |
| `lung_prediction_failures_total` | Failed predictions |
| `lung_prediction_latency_seconds` | Prediction latency |
| `lung_model_ready` | Whether the model loaded successfully |
| `lung_app_uptime_seconds` | API uptime |

Prometheus scrapes these metrics, and Grafana displays them in the provisioned dashboard.

## Docker Services

| Compose Service | Purpose | Host Port |
|-----------------|---------|-----------|
| `api` | FastAPI prediction API | 8000 |
| `frontend` | Static web application | 8001 |
| `mlflow` | MLflow experiment UI | 5000 |
| `model_server` | MLflow model serving | 5001 |
| `airflow` | Airflow standalone server | 8080 |
| `prometheus` | Metrics collection | 9090 |
| `grafana` | Metrics dashboards | 3001 |
| `alertmanager` | Alert handling | 9093 |

Useful commands:

```bash
sudo docker-compose ps
sudo docker-compose logs airflow --tail=120
sudo docker-compose logs api --tail=120
sudo docker-compose down --remove-orphans
sudo ./scripts/compose-up-clean.sh
```

## Troubleshooting

### Port Is Already in Use

If Docker says `bind: address already in use`, another service is already using that port.

Find the process:

```bash
sudo lsof -i :8080
sudo lsof -i :3001
```

Stop the process or change the host port in `docker-compose.yml`.

### Docker Compose Shows `KeyError: 'ContainerConfig'`

This can happen with `docker-compose` v1.29.2 and newer Docker versions when recreating containers.

Fix:

```bash
sudo ./scripts/compose-up-clean.sh
```

### Airflow Does Not Start

Check logs:

```bash
sudo docker-compose logs airflow --tail=200
```

Then rebuild only Airflow:

```bash
sudo docker-compose build --no-cache airflow
sudo docker-compose up --force-recreate airflow
```

### API Says Model Is Not Ready

Check that this folder exists:

```text
models/latest/
```

If it is missing, train the model:

```bash
python src/models/train.py
```

Then restart the API container:

```bash
sudo docker-compose restart api
```

### Grafana Dashboard Is Empty

First check the API metrics:

```text
http://localhost:8000/metrics
```

Then check Prometheus targets:

```text
http://localhost:9090/targets
```

If `lung-api` is down, inspect the API logs:

```bash
sudo docker-compose logs api --tail=120
```


## Summary

This repository demonstrates a complete beginner-friendly MLOps workflow:

1. Train a PyTorch image classifier.
2. Track experiments with MLflow.
3. Serve predictions with FastAPI.
4. Use a simple browser UI.
5. Orchestrate training with Airflow.
6. Monitor production behavior with Prometheus and Grafana.
7. Run the whole stack with Docker Compose.
