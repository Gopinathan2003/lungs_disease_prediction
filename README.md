# Lung Disease Prediction using Chest X-Rays | MLOps Project

An end-to-end AI application for detecting lung diseases from Chest X-Ray images with full MLOps implementation. Built with emphasis on automation, reproducibility, monitoring, and production readiness.

---

## 📋 Project Overview

This project demonstrates a complete AI Product Lifecycle following strict MLOps principles:

- Automated data pipeline
- Reproducible model training & experimentation
- Containerized deployment (Docker + Docker Compose)
- Real-time model serving
- Monitoring with Prometheus, Grafana & Alertmanager
- Experiment tracking with MLflow
- Version control using Git + DVC
- Pipeline orchestration with Apache Airflow

### Problem Statement
Multi-class classification of Chest X-Ray images into:
- Normal
- Bacterial Pneumonia
- Viral Pneumonia
- Tuberculosis
- Corona Virus Disease

### Key Metrics
| Metric | Target |
|--------|--------|
| ML Metrics | F1-Score ≥ 0.90, Accuracy ≥ 0.92 |
| Business Metrics | Inference latency < 200ms |

---

## 🛠️ Technology Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.11 |
| Model | ResNet18 (Transfer Learning) |
| Experiment Tracking | MLflow |
| Pipeline | DVC + Airflow |
| API | FastAPI |
| Model Serving | MLflow Model Server |
| Frontend | HTML + Bootstrap + JavaScript |
| Monitoring | Prometheus + Grafana + Alertmanager |
| Version Control | Git + DVC + Git LFS |
| Orchestration | Apache Airflow |
| Containerization | Docker + Docker Compose |

---

## 📁 Project Structure

```
lungs_disease_prediction/
├── data/
│   ├── raw/                    # Raw chest X-ray dataset
│   │   └── chest_xray/
│   │       ├── train/
│   │       ├── test/
│   │       └── val/
│   └── processed/              # Preprocessed data + metadata
│       ├── train/
│       ├── test/
│       └── val/
├── src/
│   ├── data/                   # Data ingestion
│   │   └── ingestion.py
│   ├── features/               # Feature engineering + baseline stats
│   │   └── build_features.py
│   ├── models/                 # Training script
│   │   └── train.py
│   └── api/                    # FastAPI backend
│       └── main.py
├── frontend/                   # Static web UI
│   └── index.html
├── airflow/
│   ├── dags/                   # Airflow pipelines
│   │   └── ml_pipeline.py
│   └── config/
│       └── airflow.cfg
├── prometheus/                 # Prometheus configuration
│   ├── prometheus.yml
│   └── alerts.yml
├── grafana/                    # Grafana dashboards
│   └── provisioning/
│       ├── dashboards/
│       └── datasources/
├── alertmanager/               # Alertmanager configuration
│   └── alertmanager.yml
├── models/                     # Trained model artifacts
│   └── latest/
├── mlruns/                     # MLflow experiment logs
├── tests/                      # Unit and integration tests
├── docs/                       # Documentation
│   ├── architecture.md
│   ├── HLD.md
│   ├── LLD.md
│   ├── test_plan.md
│   ├── user_manual.md
│   └── project_report.md
├── dvc.yaml                    # DVC pipeline definition
├── MLProject                   # MLflow project definition
├── docker-compose.yml          # Docker Compose orchestration
├── dockerfile                  # Docker image definition
├── requirements.txt            # Python dependencies
├── requirements-serving.txt    # Model serving dependencies
└── README.md
```

---

## 🚀 Quick Start

### Option 1: Using Docker Compose (Recommended)

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd lungs_disease_prediction
   ```

2. **Setup Environment Variables**
   ```bash
   cp .env.example .env
   # Fill in the Mailtrap SMTP credentials for email notifications
   ```

3. **Train the Model**
   ```bash
   python src/models/train.py
   ```
   Or use Airflow DAG: `http://localhost:8088`

4. **Start All Services**
   ```bash
   docker compose up --build
   ```

### Option 2: Without Docker

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd lungs_disease_prediction
   ```

2. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/Mac
   # venv\Scripts\activate         # Windows

   pip install -r requirements.txt
   ```

3. **Download Dataset**
   Download the Chest X-Ray Images dataset from Kaggle and place it in:
   ```
   data/raw/chest_xray/
   ```

4. **Run DVC Pipeline**
   ```bash
   dvc repro
   ```

5. **Train the Model**
   ```bash
   python src/models/train.py
   ```
   Copy the Run ID shown after training.

6. **Start Services (in separate terminals)**

   **Terminal 1: MLflow UI**
   ```bash
   mlflow ui --host 0.0.0.0 --port 5000
   ```

   **Terminal 2: MLflow Model Server**
   ```bash
   mlflow models serve -m "runs:/<YOUR_RUN_ID_HERE>/model" -p 8080 --host 0.0.0.0 --no-conda
   ```

   **Terminal 3: FastAPI Backend**
   ```bash
   cd src/api
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

   **Terminal 4: Frontend**
   ```bash
   cd frontend
   python -m http.server 8081
   ```

   **Terminal 5: Airflow (Optional)**
   ```bash
   airflow webserver -p 8088
   ```

---

## 🌐 Access Points

| Service | URL | Purpose |
|---------|-----|---------|
| Web Application | http://localhost:8081 | User Interface |
| API Documentation | http://localhost:8000/docs | Swagger UI |
| MLflow Dashboard | http://localhost:5000 | Experiment Tracking |
| Prometheus | http://localhost:9090 | Metrics & Alerts |
| Grafana | http://localhost:3000 | Visualization & Alerts |
| Airflow | http://localhost:8088 | Pipeline Orchestration |
| Alertmanager | http://localhost:9093 | Alert Management |

---

## 📊 How to Test the Application

1. **Open the Web UI** → http://localhost:8081
2. **Upload a Chest X-Ray image** (JPEG/PNG)
3. **Click Predict**
4. **View prediction** (Normal / Pneumonia / Tuberculosis / Corona Virus), confidence, and latency
5. **Check MLflow** for experiment logs at http://localhost:5000
6. **Monitor metrics** at http://localhost:8000/metrics
7. **View dashboards** in Grafana at http://localhost:3000

---

## 🧪 Running Tests

```bash
# Run unit tests
python -m pytest tests/

# View test report
cat docs/test_report.md
```

---

## 📚 Documentation

All required documents are available in the `/docs` folder:

| Document | Description |
|----------|-------------|
| [architecture.md](docs/architecture.md) | Architecture Diagram with explanation |
| [HLD.md](docs/HLD.md) | High-Level Design document |
| [LLD.md](docs/LLD.md) | Low-Level Design with API endpoints |
| [test_plan.md](docs/test_plan.md) | Test Plan & Test Cases |
| [user_manual.md](docs/user_manual.md) | User Manual (for non-technical users) |
| [project_report.md](docs/project_report.md) | Complete Project Report |

---

## 🔄 MLOps Features Implemented

| Feature | Implementation |
|---------|----------------|
| ✅ Reproducibility | Git commit hash + MLflow Run ID |
| ✅ Automation | DVC pipeline for data & features |
| ✅ Experiment Tracking | MLflow (metrics, params, artifacts, models) |
| ✅ Model Versioning | MLflow Model Registry |
| ✅ Data Versioning | DVC |
| ✅ Monitoring | Prometheus instrumentation + Grafana dashboards |
| ✅ Alerting | Alertmanager integration |
| ✅ CI/CD Pipeline | DVC DAG + Airflow |
| ✅ API | FastAPI with health & ready endpoints |
| ✅ Loose Coupling | Frontend and Backend connected only via REST API |
| ✅ Drift Baseline | Statistical baselines calculated during EDA |
| ✅ Containerization | Docker + Docker Compose |
| ✅ Pipeline Orchestration | Apache Airflow DAG |

---

## 📈 Monitoring

### Prometheus Metrics
Prometheus metrics are exposed at:
```
http://localhost:8000/metrics
```

### Key Metrics Tracked
- `total_predictions` - Total number of predictions made
- `prediction_latency_seconds` - Inference latency histogram
- `http_requests_total` - HTTP request count by status
- `model_ready` - Model availability status
- `prediction_failures_total` - Failed prediction count

### Grafana Dashboards
Pre-configured dashboards for:
- API Performance
- Model Metrics
- Prediction Statistics
- System Health

### Alertmanager Alerts
Configured alerts for:
- API downtime
- Model readiness issues
- Prediction failures
- High latency (>200ms)

---

## 🐳 Docker Services

The `docker-compose.yml` includes the following services:

| Service | Description | Port |
|---------|-------------|------|
| api | FastAPI application | 8000 |
| frontend | Web UI | 8081 |
| mlflow | MLflow tracking server | 5000 |
| prometheus | Metrics collection | 9090 |
| grafana | Visualization | 3000 |
| airflow-webserver | Airflow UI | 8088 |
| airflow-scheduler | Airflow scheduler | - |
| alertmanager | Alert management | 9093 |

---
