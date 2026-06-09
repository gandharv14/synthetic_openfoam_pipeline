#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math


def _clip(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(value)))


def _plume_push(obs):
    x = float(obs["x"])
    z = float(obs["z"])
    radius = float(obs.get("glider_radius", 0.03))
    push_x = 0.0
    push_z = 0.0
    for plume in obs.get("plumes", []):
        cx, cz = plume["center"]
        dx = x - float(cx)
        dz = z - float(cz)
        dist = max(1e-6, math.hypot(dx, dz))
        clearance = dist - float(plume["radius"]) - radius
        influence = 0.24
        if clearance < influence:
            gain = ((influence - clearance) / influence) ** 2
            push_x += gain * dx / dist
            push_z += gain * dz / dist
    return push_x, push_z


def _wall_push(obs):
    x = float(obs["x"])
    z = float(obs["z"])
    ws = obs.get("workspace", {})
    radius = float(obs.get("glider_radius", 0.03))
    margin = 0.16
    push_z = 0.0
    upper = z - float(ws.get("z_min", 0.2)) - radius
    lower = float(ws.get("z_max", 1.4)) - z - radius
    if upper < margin:
        push_z += (margin - upper) / margin
    if lower < margin:
        push_z -= (margin - lower) / margin
    return push_z


def act(obs):
    x = float(obs["x"])
    z = float(obs["z"])
    vx = float(obs["vx"])
    vz = float(obs["vz"])
    pitch = float(obs.get("pitch", 0.0))
    current_x = float(obs.get("current_x", 0.0))
    current_z = float(obs.get("current_z", 0.0))
    goal_x = float(obs["goal_x"])
    goal_z = float(obs["goal_z"])
    target_depth = float(obs.get("target_depth", goal_z))
    slope = float(obs.get("target_slope", 0.0))

    # Lead the target in x because this glider cannot stop and reverse.
    lead = _clip(0.12 + 0.40 * max(0.0, goal_x - x), 0.10, 0.30)
    profile_goal = target_depth + slope * lead
    if obs.get("goal_kind") == "sample":
        desired_z = 0.82 * goal_z + 0.18 * profile_goal
    else:
        desired_z = goal_z

    _, plume_z = _plume_push(obs)
    wall_z = _wall_push(obs)
    desired_z += 0.13 * plume_z + 0.10 * wall_z

    depth_error = desired_z - z
    goal_dx = max(0.04, goal_x - x)
    desired_vz = _clip(2.35 * depth_error + 0.20 * slope * goal_dx - 1.05 * current_z, -0.34, 0.34)
    desired_pitch = _clip(math.asin(_clip(desired_vz / 0.44, -0.92, 0.92)), -0.62, 0.62)
    pitch_cmd = _clip(3.4 * (desired_pitch - pitch))

    buoyancy_cmd = _clip(3.60 * (desired_vz - vz) + 1.35 * depth_error - 0.75 * current_z + 0.10 * plume_z + 0.10 * wall_z)
    if x > float(obs.get("finish_x", 2.0)) - 0.12 and obs.get("goal_kind") == "finish":
        buoyancy_cmd = _clip(2.9 * (goal_z - z) - 1.2 * vz - 0.8 * current_z)
        pitch_cmd = _clip(2.2 * (desired_pitch - pitch) - 0.3 * (vx + current_x - 0.16))
    return [pitch_cmd, buoyancy_cmd]


def get_action(obs):
    return act(obs)
PY

cat > "${OUTPUT_DIR}/README.md" <<'MD'
Predictive depth controller for a buoyancy/pitch glider, using current cancellation plus wall and plume repulsion.
MD
