from datetime import datetime
from pathlib import Path

from airflow import DAG
from airflow.operators.bash import BashOperator

PROJECT_ROOT = Path("/opt/airflow/project")

with DAG(
    dag_id="lung_disease_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "mlflow", "lungs"],
) as dag:
    ingest = BashOperator(
        task_id="ingest_data",
        bash_command=f"cd {PROJECT_ROOT} && python src/data/ingestion.py",
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=f"cd {PROJECT_ROOT} && python src/features/build_features.py",
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=f"cd {PROJECT_ROOT} && python src/models/train.py",
    )

    ingest >> build_features >> train_model
