import os
from datetime import datetime
from pathlib import Path

from airflow import DAG

try:
    from airflow.providers.standard.operators.bash import BashOperator
except ImportError:
    from airflow.operators.bash import BashOperator

DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(os.getenv("PROJECT_ROOT", str(DEFAULT_PROJECT_ROOT))).resolve()


def project_command(module_path: str) -> str:
    return (
        "set -e\n"
        f"cd {PROJECT_ROOT}\n"
        f"export PYTHONPATH={PROJECT_ROOT}:$PYTHONPATH\n"
        f"python -m {module_path}"
    )


with DAG(
    dag_id="lung_disease_training_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
    tags=["mlops", "mlflow", "lungs"],
) as dag:
    ingest = BashOperator(
        task_id="ingest_data",
        bash_command=project_command("src.data.ingestion"),
    )

    build_features = BashOperator(
        task_id="build_features",
        bash_command=project_command("src.features.build_features"),
    )

    train_model = BashOperator(
        task_id="train_model",
        bash_command=project_command("src.models.train"),
    )

    ingest >> build_features >> train_model
