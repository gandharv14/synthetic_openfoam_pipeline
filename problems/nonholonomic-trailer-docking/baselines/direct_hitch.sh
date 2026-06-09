#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clip(value):
    return max(-1.0, min(1.0, float(value)))


def act(obs):
    dx = obs["desired_hitch_x"] - obs["hitch_x"]
    dy = obs["desired_hitch_y"] - obs["hitch_y"]
    desired = math.atan2(dy, dx)
    err = _wrap(desired - obs["tractor_yaw"])
    # This intentionally moves the tractor hitch to the target and ignores
    # trailer yaw, jackknife recovery, and no-go layout.
    drive = math.hypot(dx, dy) * math.cos(err)
    steer = 1.4 * err
    return [_clip(drive), _clip(steer)]
PY
