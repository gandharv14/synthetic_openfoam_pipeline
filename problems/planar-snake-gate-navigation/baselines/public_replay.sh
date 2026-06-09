#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math

PUBLIC_POINTS = [
    [-0.46, 0.08],
    [-0.04, -0.10],
    [0.39, 0.08],
    [0.78, 0.12],
]


def _wrap(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def act(obs):
    head_x, head_y = obs.get("head_xy", [0.0, 0.0])
    yaw = float(obs.get("head_yaw", 0.0))
    target = PUBLIC_POINTS[min(int(obs.get("gate_index", 0)), len(PUBLIC_POINTS) - 1)]
    err = _wrap(math.atan2(target[1] - head_y, target[0] - head_x) - yaw)
    phase = 2.0 * math.pi * 0.75 * float(obs.get("time", 0.0))
    joints = [0.32 * math.sin(phase - 0.7 * i) for i in range(int(obs.get("num_joints", 6)))]
    return [max(-0.2, min(0.75, 0.70 * math.cos(err))), max(-1.0, min(1.0, err)), *joints]
PY
