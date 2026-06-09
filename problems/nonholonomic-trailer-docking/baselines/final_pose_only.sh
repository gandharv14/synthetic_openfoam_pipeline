#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _clip(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(value)))


def act(obs):
    # A naive final-pose controller: tries to fix trailer yaw and drive toward
    # the target, but ignores hitch-angle stability and obstacles.
    theta = obs["trailer_yaw"]
    ux, uy = math.cos(theta), math.sin(theta)
    longitudinal = obs["target_dx"] * ux + obs["target_dy"] * uy
    drive = min(0.25, 1.2 * longitudinal)
    if obs.get("reverse_required", False):
        drive = min(drive, -0.2)
    desired_phi = _clip(1.0 * obs["target_yaw_error"], -0.55, 0.55)
    steer = 1.6 * _wrap(theta - desired_phi - obs["tractor_yaw"])
    return [_clip(drive), _clip(steer)]
PY
