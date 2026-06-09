#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
import math


def _clip(value, lo=-1.0, hi=1.0):
    return max(lo, min(hi, float(value)))


def _wrap(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _unit(yaw):
    return math.cos(yaw), math.sin(yaw)


def _clearance_penalty(obs, hx, hy, tractor_yaw, trailer_yaw):
    length = float(obs["trailer_length"])
    ux, uy = _unit(trailer_yaw)
    tx, ty = _unit(tractor_yaw)
    points = [
        (hx, hy),
        (hx + 0.42 * tx, hy + 0.42 * ty),
        (hx - 0.5 * length * ux, hy - 0.5 * length * uy),
        (hx - length * ux, hy - length * uy),
    ]
    penalty = 0.0
    workspace = obs.get("workspace", {})
    for px, py in points:
        margin = min(
            px - float(workspace.get("x_min", -1.55)) - 0.08,
            float(workspace.get("x_max", 1.60)) - px - 0.08,
            py - float(workspace.get("y_min", -1.05)) - 0.08,
            float(workspace.get("y_max", 1.05)) - py - 0.08,
        )
        if margin < 0.04:
            penalty += 14.0 * (0.04 - margin) ** 2 + (1.0 if margin < 0.0 else 0.0)
        for region in obs.get("no_go", []):
            if region.get("type") != "circle":
                continue
            cx, cy = region["center"]
            clearance = math.hypot(px - float(cx), py - float(cy)) - float(region["radius"]) - 0.08
            if clearance < 0.05:
                penalty += 18.0 * (0.05 - clearance) ** 2 + (1.2 if clearance < 0.0 else 0.0)
    return penalty


def _rollout_cost(obs, drive_cmd, steer_cmd):
    hx = float(obs["hitch_x"])
    hy = float(obs["hitch_y"])
    psi = float(obs["tractor_yaw"])
    phi = float(obs["hitch_angle"])
    length = float(obs["trailer_length"])
    dt = float(obs["dt"])
    max_drive = float(obs["max_drive_speed"])
    max_yaw = float(obs["max_yaw_rate"])
    remaining = max(0.0, float(obs["duration"]) - float(obs["time"]))
    horizon = max(6, min(28, int(min(0.62, remaining) / max(dt, 1e-6))))

    cost = 0.0
    for _ in range(horizon):
        theta = _wrap(psi + phi)
        v = drive_cmd * max_drive
        yaw_rate = steer_cmd * max_yaw
        theta_dot = -(v / max(length, 1e-6)) * math.sin(phi)
        phi_dot = theta_dot - yaw_rate
        hx += v * math.cos(psi) * dt
        hy += v * math.sin(psi) * dt
        psi = _wrap(psi + yaw_rate * dt)
        phi = _wrap(phi + phi_dot * dt)
        theta = _wrap(psi + phi)
        if abs(phi) > 1.05:
            cost += 0.35 + 1.8 * (abs(phi) - 1.05) ** 2
        cost += 0.10 * _clearance_penalty(obs, hx, hy, psi, theta)

    theta = _wrap(psi + phi)
    ux, uy = _unit(theta)
    cx = hx - 0.5 * length * ux
    cy = hy - 0.5 * length * uy
    pos_error = math.hypot(float(obs["target_x"]) - cx, float(obs["target_y"]) - cy)
    yaw_error = abs(_wrap(float(obs["target_yaw"]) - theta))
    desired_hitch_error = math.hypot(float(obs["desired_hitch_x"]) - hx, float(obs["desired_hitch_y"]) - hy)
    final_weight = 1.0 + max(0.0, 1.5 - remaining)
    cost += final_weight * (4.0 * pos_error + 1.55 * yaw_error + 0.60 * desired_hitch_error)
    cost += 0.95 * abs(phi) + 0.08 * abs(drive_cmd) + 0.04 * abs(steer_cmd)
    cost += _clearance_penalty(obs, hx, hy, psi, theta)
    if obs.get("reverse_required", False) and remaining > 1.0 and drive_cmd > 0.05 and abs(phi) < 0.78:
        cost += 0.85 + 0.5 * drive_cmd
    if remaining < 1.0:
        cost += 0.70 * abs(drive_cmd)
    return cost


def _heuristic_action(obs):
    tx = obs["target_x"]
    ty = obs["target_y"]
    theta = obs["trailer_yaw"]
    psi = obs["tractor_yaw"]
    phi = obs["hitch_angle"]
    dx = tx - obs["trailer_x"]
    dy = ty - obs["trailer_y"]
    ux, uy = _unit(theta)
    nx, ny = -uy, ux
    longitudinal = dx * ux + dy * uy
    lateral = dx * nx + dy * ny
    yaw_error = _wrap(obs["target_yaw"] - theta)
    if abs(phi) > 0.82 and obs["time"] < obs["duration"] - 1.3:
        drive = 0.34
        desired_tractor_yaw = theta
    else:
        drive = _clip(1.35 * longitudinal, -0.84, 0.30)
        if obs.get("reverse_required", False):
            drive = min(drive, -0.16)
        desired_phi = _clip(1.05 * yaw_error + 1.20 * lateral, -0.58, 0.58)
        desired_tractor_yaw = _wrap(theta - desired_phi)
    steer = 1.75 * _wrap(desired_tractor_yaw - psi) - 0.18 * obs["hitch_angle_rate"]
    return [_clip(drive), _clip(steer)]


def act(obs):
    base_drive, base_steer = _heuristic_action(obs)
    drive_values = [-0.92, -0.70, -0.48, -0.28, -0.12]
    if abs(obs["hitch_angle"]) > 0.78:
        drive_values += [0.16, 0.34]
    elif obs["duration"] - obs["time"] < 1.2:
        drive_values += [0.0, 0.12]
    steer_values = [-1.0, -0.65, -0.35, 0.0, 0.35, 0.65, 1.0]
    candidates = [(base_drive, base_steer)]
    candidates.extend((d, s) for d in drive_values for s in steer_values)

    best_action = candidates[0]
    best_cost = float("inf")
    for drive, steer in candidates:
        cost = _rollout_cost(obs, _clip(drive), _clip(steer))
        if cost < best_cost:
            best_cost = cost
            best_action = (_clip(drive), _clip(steer))
    return [best_action[0], best_action[1]]
PY

cat > "${OUTPUT_DIR}/README.md" <<'MD'
Deterministic staged trailer controller: recover hitch angle, back along the
trailer axis with lateral/yaw feedback, add small no-go repulsion, and damp the
final hold.
MD
