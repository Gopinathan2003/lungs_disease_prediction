# lungs_disease_prediction
This project aims to develop an AI-based app to predict lung diseases such as Pneumonia, Tuberculosis, and Chronic obstructive pulmonary disease using medical datasets containing patient symptoms or clinical data. The system follows MLOps practices including automated data preprocessing, model training, experiment tracking with MLflow, orchestration with Airflow, observability with Prometheus, and alerting with Alertmanager.

## What was added
- Airflow DAG at `airflow/dags/ml_pipeline.py` to orchestrate ingestion, feature engineering, and training.
- MLflow training instrumentation that logs per-epoch train/test `loss`, `accuracy`, and `f1_score`, plus a comparison plot artifact.
- Prometheus metrics on the FastAPI app under `/metrics` for readiness, request outcomes, latency, and prediction failures.
- Alertmanager integration with Prometheus alert rules for API downtime, model readiness issues, prediction failures, and high latency.
- Mailtrap SMTP placeholders in `.env.example` for email notifications.

## Run the stack
1. Copy `.env.example` to `.env` and fill in the Mailtrap SMTP credentials.
2. Train once locally or through Airflow so `models/latest` exists.
3. Start the services with `docker compose up --build`.

## Ports
- API: `8000`
- Frontend: `8081`
- Prometheus: `9090`
- Grafana: `3000`
- Alertmanager: `9093`
- Airflow: `8088`
