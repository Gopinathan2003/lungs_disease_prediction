# Lung Disease Prediction Project Report

## Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [High-Level Design (HLD)](#high-level-design-hld)
4. [Low-Level Design (LLD)](#low-level-design-lld)
5. [Working Flow](#working-flow)
6. [Test Plan and Cases](#test-plan-and-cases)
7. [Test Report](#test-report)
8. [User Manual](#user-manual)
9. [Deployment Run Book](#deployment-run-book)
10. [Screenshots](#screenshots)

---

## Project Overview
This project is a Lung Disease Predictor that leverages deep learning (ResNet18) to classify chest X-ray images into multiple disease categories. It is built with an MLOps pipeline using Airflow, DVC, MLflow, Prometheus, and Grafana for end-to-end automation, monitoring, and reproducibility.

---

## Architecture
- **Frontend UI**: User uploads X-ray images and receives predictions.
- **FastAPI Service**: REST API for inference requests.
- **Model Server**: MLflow/TorchServe serving the trained PyTorch ResNet18 model.
- **Airflow DAG**: Orchestrates data ingestion, preprocessing, model training, and deployment.
- **DVC**: Manages data and model versioning.
- **MLflow**: Tracks experiments, metrics, and model registry.
- **Prometheus & Grafana**: Monitors API, model, and pipeline health.

---

## High-Level Design (HLD)
- **Data Flow**: Raw data → Preprocessing → Feature Engineering → Model Training → Model Registry → Model Serving
- **Pipeline Orchestration**: Airflow DAG manages the end-to-end workflow.
- **Monitoring**: Prometheus scrapes metrics from API/model, Grafana visualizes them, and Alertmanager triggers alerts.

---

## Low-Level Design (LLD)
- **Data Ingestion**: Scripts in `src/data/ingestion.py` handle loading and preprocessing.
- **Feature Engineering**: `src/features/build_features.py` extracts features.
- **Model Training**: `src/models/train.py` trains and logs models to MLflow.
- **API**: `src/api/main.py` exposes endpoints for predictions.
- **Monitoring**: Custom metrics are exposed for Prometheus scraping.

---

## Working Flow
![Working Flow Diagram](docs/working_flow.png)

*The above diagram illustrates the end-to-end workflow, from data ingestion to monitoring and user interaction.*

---

## Test Plan and Cases
### Test Plan
- **Unit Tests**: For data processing, feature engineering, and model logic.
- **Integration Tests**: For API endpoints and pipeline steps.
- **E2E Tests**: For the full prediction flow from UI to model server.

### Test Cases
| Test Case ID | Description | Expected Result |
|--------------|-------------|----------------|
| TC-01 | Upload valid X-ray image | Correct disease prediction |
| TC-02 | Upload invalid file | Error message |
| TC-03 | API returns prediction | 200 OK with result |
| TC-04 | Model not ready | 503 Service Unavailable |
| TC-05 | High latency alert | Alert triggered in Prometheus |

---

## Test Report
- **Unit Tests**: All passed (see `tests/` folder for details).
- **Integration Tests**: API and pipeline integration verified.
- **E2E Tests**: UI to model server flow validated.
- **Monitoring**: Prometheus alerts tested (see screenshot).

---

## User Manual
### 1. Uploading an Image
- Open the Frontend UI at `http://localhost:8081`.
- Click 'Browse...' to select a chest X-ray image.
- Click 'Predict' to get the disease prediction, confidence, and latency.

### 2. Viewing Metrics and Monitoring
- **MLflow**: Visit `http://127.0.0.1:5000` for experiment tracking.
- **Prometheus**: Visit `http://localhost:9090` for metrics and alerts.
- **Grafana**: Visit the Grafana dashboard for visual monitoring.

### 3. Running the Pipeline
- Use Airflow UI to trigger and monitor DAG runs.
- Data and model versions are managed by DVC.

---

## Deployment Run Book
1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Start services**: Use `docker-compose up` for all services (API, model server, monitoring, etc.)
4. **Access UIs**:
   - Frontend: `http://localhost:8081`
   - API: `http://localhost:8000`
   - MLflow: `http://127.0.0.1:5000`
   - Prometheus: `http://localhost:9090`
   - Grafana: (configured port)
5. **Run Airflow DAG**: Trigger from Airflow UI or CLI.
6. **Monitor system**: Use Prometheus and Grafana dashboards.

---

## Screenshots
### 1. Frontend UI
![Frontend UI](docs/screenshots/frontend_ui.png)

### 2. Prometheus Alerts
![Prometheus Alerts](docs/screenshots/prometheus_alerts.png)

### 3. MLflow Metrics
![MLflow Metrics](docs/screenshots/mlflow_metrics.png)

### 4. Working Flow Diagram
![Working Flow](docs/working_flow.png)

---

*For more details, refer to the README.md and code documentation.*
