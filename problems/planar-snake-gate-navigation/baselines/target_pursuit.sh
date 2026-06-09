#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"
cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math


def _wrap(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def act(obs):
    head_x, head_y = obs.get("head_xy", [0.0, 0.0])
    yaw = float(obs.get("head_yaw", 0.0))
    gate = obs.get("target_gate") or {}
    target_x, target_y = gate.get("center", obs.get("final_target", [0.0, 0.0]))
    err = _wrap(math.atan2(target_y - head_y, target_x - head_x) - yaw)
    drive = max(-0.15, min(0.78, 0.72 * math.cos(err)))
    turn = max(-1.0, min(1.0, 1.25 * err))
    return [drive, turn, *([0.0] * int(obs.get("num_joints", 6)))]
PY
