"""Oracle PD recovery controller for the roly-poly upright recovery task."""

from __future__ import annotations

import math


class Policy:
    def __init__(self) -> None:
        self._phase = "recover"

    def act(self, obs: dict) -> float:
        tilt = float(obs["tilt_angle"])
        rate = float(obs["tilt_vel"])
        upright = float(obs["upright_z"])
        t_frac = float(obs["time"]) / max(1e-6, float(obs["duration"]))
        friction = float(obs.get("floor_friction", 1.0))
        mass_off = float(obs.get("mass_offset", 0.0))
        damp_scale = float(obs.get("damping_scale", 1.0))
        lateral = float(obs.get("com_lateral_offset", 0.0))

        kp = 58.0 + 8.0 * friction + 14.0 * mass_off
        kd = (12.0 + 4.0 * damp_scale) * (1.0 + 0.25 * friction)
        kp *= 1.0 + 0.4 * abs(lateral)
        kd *= 1.0 + 0.15 * abs(lateral)

        target = -0.35 * lateral
        u = kp * (target - tilt) - kd * rate

        if upright < 0.55 and t_frac < 0.55:
            u += 3.8 * math.copysign(1.0, -tilt) * (0.55 - upright)
        if upright >= 0.9 and abs(tilt) < 0.12:
            self._phase = "hold"
        if self._phase == "hold":
            u = 0.65 * kp * (target - tilt) - (1.15 * kd) * rate

        settle = min(1.0, max(0.0, (t_frac - 0.5) / 0.35))
        u *= 0.85 + 0.15 * settle
        return float(max(-12.0, min(12.0, u)))


_ORACLE = Policy()


def act(obs):
    if isinstance(obs, dict):
        return _ORACLE.act(obs)
    return _ORACLE.act({"tilt_angle": 0.0, "tilt_vel": 0.0, "upright_z": 1.0, "time": 0.0, "duration": 1.0})
