#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
def act(obs):
    z_error = float(obs["goal_z"]) - float(obs["z"])
    return [0.0, max(-1.0, min(1.0, 2.0 * z_error))]
PY
