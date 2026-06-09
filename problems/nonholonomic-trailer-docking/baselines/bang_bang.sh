#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math


def act(obs):
    drive = -1.0 if obs.get("reverse_required", False) else 1.0
    steer = 1.0 if obs["target_yaw_error"] + obs["target_dy"] > 0.0 else -1.0
    return [drive, steer]
PY
