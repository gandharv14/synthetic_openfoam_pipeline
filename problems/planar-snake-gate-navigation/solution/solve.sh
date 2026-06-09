#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
from __future__ import annotations

import math


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


class Policy:
    def act(self, obs):
        head = obs.get("head_xy", [0.0, 0.0])
        yaw = float(obs.get("head_yaw", 0.0))
        velocity_body = obs.get("head_velocity_body", [0.0, 0.0])
        gate = obs.get("target_gate") or {}
        next_gate = obs.get("next_gate")
        target = list(gate.get("center", obs.get("final_target", [0.0, 0.0])))
        gate_dx = target[0] - head[0]
        gate_dy = target[1] - head[1]
        gate_dist = math.hypot(gate_dx, gate_dy)

        if next_gate is not None and gate_dist < 0.24:
            nx, ny = next_gate.get("center", target)
            target[0] = 0.45 * target[0] + 0.55 * float(nx)
            target[1] = 0.45 * target[1] + 0.55 * float(ny)

        desired_x = target[0] - head[0]
        desired_y = target[1] - head[1]

        for item in obs.get("no_go", []):
            if item.get("type") != "circle":
                continue
            cx, cy = item.get("center", [0.0, 0.0])
            radius = float(item.get("radius", 0.1))
            away_x = head[0] - float(cx)
            away_y = head[1] - float(cy)
            dist = max(1e-4, math.hypot(away_x, away_y))
            influence = radius + 0.30
            if dist < influence:
                gain = 0.38 * (influence - dist) / influence
                desired_x += gain * away_x / dist
                desired_y += gain * away_y / dist

        workspace = obs.get("workspace", {})
        margin = 0.18
        if "x_min" in workspace and head[0] < float(workspace["x_min"]) + margin:
            desired_x += 0.35
        if "x_max" in workspace and head[0] > float(workspace["x_max"]) - margin:
            desired_x -= 0.35
        if "y_min" in workspace and head[1] < float(workspace["y_min"]) + margin:
            desired_y += 0.35
        if "y_max" in workspace and head[1] > float(workspace["y_max"]) - margin:
            desired_y -= 0.35

        desired_heading = math.atan2(desired_y, desired_x)
        heading_error = _wrap(desired_heading - yaw)
        distance = math.hypot(desired_x, desired_y)
        forward_speed = float(velocity_body[0]) if velocity_body else 0.0
        drive = _clip(0.74 * distance + 0.22 * math.cos(heading_error) - 0.26 * forward_speed, -0.20, 0.92)
        if abs(heading_error) > 1.35:
            drive *= 0.25
        elif abs(heading_error) > 0.78:
            drive *= 0.55
        turn = _clip(1.38 * heading_error)

        num_joints = int(obs.get("num_joints", 6))
        phase = 2.0 * math.pi * (0.86 * float(obs.get("time", 0.0)))
        amplitude = 0.34 + 0.12 * min(1.0, abs(heading_error))
        bend_bias = _clip(0.23 * heading_error, -0.26, 0.26)
        joints = []
        for idx in range(num_joints):
            wave = amplitude * math.sin(phase - 0.74 * idx)
            bias = bend_bias * math.exp(-0.25 * idx)
            joints.append(_clip(wave + bias, -0.92, 0.92))
        return [drive, turn, *joints]


_POLICY = Policy()


def act(obs):
    return _POLICY.act(obs)
PY
