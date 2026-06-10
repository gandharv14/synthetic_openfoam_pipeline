#!/usr/bin/env bash
set -euo pipefail
OUT="${LBT_OUTPUT_DIR:-/tmp/output}"
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${LBT_DATA_DIR:-/data}"
if [[ ! -f "${DATA_DIR}/starter_case.tar.gz" ]]; then
  DATA_DIR="${TASK_DIR}/data"
fi
WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
mkdir -p "${OUT}" "${WORK}/case"
tar -xzf "${DATA_DIR}/starter_case.tar.gz" -C "${WORK}/case" --strip-components=1
tar -czf "${OUT}/fixed_case.tar.gz" -C "${WORK}" case
