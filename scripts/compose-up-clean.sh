#!/usr/bin/env bash
set -euo pipefail

COMPOSE="${COMPOSE:-docker-compose}"

if ! command -v "${COMPOSE}" >/dev/null 2>&1; then
  echo "Could not find '${COMPOSE}' on PATH." >&2
  echo "Set COMPOSE='docker compose' or install docker-compose." >&2
  exit 1
fi

# docker-compose v1.29 can hit KeyError: 'ContainerConfig' with newer Docker
# when recreating existing containers. Removing the project containers first
# avoids the broken recreate path without deleting bind-mounted project data.
"${COMPOSE}" down --remove-orphans
"${COMPOSE}" up --build --force-recreate
