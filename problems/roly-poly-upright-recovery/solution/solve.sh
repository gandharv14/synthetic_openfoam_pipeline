#!/usr/bin/env bash
set -euo pipefail

mkdir -p /tmp/output

cat > /tmp/output/model.xml <<'XML'
<?xml version="1.0"?>
<mujoco model="roly_poly_upright_recovery">
  <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81"/>
  <size njmax="120" nconmax="40"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <default>
    <geom friction="1.1 0.005 0.0001" solref="0.02 1" solimp="0.95 0.99 0.001"/>
    <joint armature="0.004" damping="0.2"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.82 0.82 0.82 1"/>
    <body name="doll" pos="0 0 0.04">
      <site name="pivot" pos="0 0 0" size="0.006" rgba="1 0.6 0.1 1"/>
      <joint name="tilt" type="hinge" axis="0 1 0" pos="0 0 0" limited="false" damping="0.22" armature="0.01"/>
      <body name="weighted_base" pos="0 0 0.01">
        <inertial pos="0 0 -0.045" mass="1.25" diaginertia="0.014 0.016 0.011"/>
        <geom name="rocker" type="sphere" size="0.078" pos="0 0 0.012" mass="1.25" rgba="0.75 0.35 0.2 1"/>
        <geom name="skirt" type="cylinder" size="0.045 0.018" pos="0 0 0.055" mass="0.0" rgba="0.55 0.28 0.18 1"/>
      </body>
      <body name="crown" pos="0 0 0.155">
        <inertial pos="0 0 0.02" mass="0.08" diaginertia="0.0007 0.0007 0.00025"/>
        <geom name="crown_geom" type="capsule" fromto="0 0 -0.03 0 0 0.07" size="0.03" mass="0.08" rgba="0.2 0.45 0.85 1"/>
        <site name="crown_top" pos="0 0 0.08" size="0.01"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="base_torque" joint="tilt" ctrlrange="-12 12" gear="1"/>
  </actuator>
  <sensor>
    <jointpos name="tilt_pos" joint="tilt"/>
    <jointvel name="tilt_vel" joint="tilt"/>
    <framezaxis name="upright_axis" objtype="body" objname="crown"/>
  </sensor>
</mujoco>
XML

cat > /tmp/output/policy.py <<'PY'
"""Oracle PD recovery controller for the roly-poly upright recovery task."""

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
    return _ORACLE.act(
        {
            "tilt_angle": 0.0,
            "tilt_vel": 0.0,
            "upright_z": 1.0,
            "time": 0.0,
            "duration": 1.0,
        }
    )
PY
