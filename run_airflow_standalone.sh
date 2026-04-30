#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export AIRFLOW_HOME="${PROJECT_ROOT}/airflow"
export AIRFLOW__CORE__DAGS_FOLDER="${PROJECT_ROOT}/airflow/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=False
export PROJECT_ROOT

AIRFLOW_PORT="${AIRFLOW_PORT:-8088}"
export AIRFLOW__API__PORT="${AIRFLOW_PORT}"
export AIRFLOW__WEBSERVER__WEB_SERVER_PORT="${AIRFLOW_PORT}"

CONDA_AIRFLOW_HOME="/home/gopuu/.conda/envs/mlenv/bin"
if [ -x "${CONDA_AIRFLOW_HOME}/airflow" ]; then
  export PATH="${CONDA_AIRFLOW_HOME}:${PATH}"
fi

AIRFLOW_BIN="${AIRFLOW_BIN:-airflow}"
if ! command -v "${AIRFLOW_BIN}" >/dev/null 2>&1; then
  AIRFLOW_BIN="${CONDA_AIRFLOW_HOME}/airflow"
fi

echo "Starting Airflow at http://localhost:${AIRFLOW_PORT}/"
"${AIRFLOW_BIN}" standalone
