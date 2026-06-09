"""Starter policy template for planar snake gate navigation."""

from __future__ import annotations

import math


def _wrap(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def act(obs: dict) -> list[float]:
    """Return [drive, turn, joint0, ..., joint5] values in roughly [-1, 1]."""

    head_x, head_y = obs["head_xy"]
    yaw = float(obs["head_yaw"])
    gate = obs["target_gate"]
    target_x, target_y = gate["center"]
    desired = math.atan2(target_y - head_y, target_x - head_x)
    heading_error = _wrap(desired - yaw)
    drive = max(-0.2, min(0.7, 0.55 * math.cos(heading_error)))
    turn = max(-1.0, min(1.0, 1.2 * heading_error))
    num_joints = int(obs.get("num_joints", 6))
    phase = 2.0 * math.pi * float(obs.get("time", 0.0))
    joints = [0.30 * math.sin(phase - 0.7 * idx) for idx in range(num_joints)]
    return [drive, turn, *joints]
