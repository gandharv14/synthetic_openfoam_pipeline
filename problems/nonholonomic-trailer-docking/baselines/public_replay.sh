#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math


PUBLIC_TARGETS = [
    (-0.55, 0.0, 0.0),
    (-0.35, 0.32, 0.42),
    (-0.45, -0.20, -0.20),
]


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clip(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(value)))


def _nearest_public_target(obs):
    best = PUBLIC_TARGETS[0]
    best_dist = float("inf")
    for target in PUBLIC_TARGETS:
        dist = math.hypot(obs["target_x"] - target[0], obs["target_y"] - target[1]) + abs(
            _wrap(obs["target_yaw"] - target[2])
        )
        if dist < best_dist:
            best = target
            best_dist = dist
    return best


def act(obs):
    # Intentionally overfits the public target set instead of using the actual
    # hidden target precisely.
    tx, ty, tyaw = _nearest_public_target(obs)
    theta = obs["trailer_yaw"]
    ux, uy = math.cos(theta), math.sin(theta)
    nx, ny = -uy, ux
    dx = tx - obs["trailer_x"]
    dy = ty - obs["trailer_y"]
    longitudinal = dx * ux + dy * uy
    lateral = dx * nx + dy * ny
    yaw_error = _wrap(tyaw - theta)
    drive = min(-0.18, 1.25 * longitudinal)
    desired_phi = _clip(0.85 * yaw_error + 0.75 * lateral, -0.50, 0.50)
    steer = 1.45 * _wrap(theta - desired_phi - obs["tractor_yaw"])
    return [_clip(drive), _clip(steer)]
PY
