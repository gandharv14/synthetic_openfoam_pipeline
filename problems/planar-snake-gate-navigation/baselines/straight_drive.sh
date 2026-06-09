#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math


def act(obs):
    n = int(obs.get("num_joints", 6))
    phase = 2.0 * math.pi * 0.8 * float(obs.get("time", 0.0))
    joints = [0.22 * math.sin(phase - 0.7 * i) for i in range(n)]
    return [0.68, 0.0, *joints]
PY
