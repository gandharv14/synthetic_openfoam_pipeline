"""Shared rollout helpers for the roly-poly upright recovery task."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import Any, Callable

import mujoco
import numpy as np

DEFAULT_DURATION = 10.0
TILT_JOINT = "tilt"
BASE_BODY = "weighted_base"
CROWN_BODY = "crown"
PIVOT_SITE = "pivot"

_MODEL_BASELINES: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}


def load_model(xml_path: Path) -> mujoco.MjModel:
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as handle:
        handle.write(xml_path.read_text())
        tmp_path = handle.name
    return mujoco.MjModel.from_xml_path(tmp_path)


def _restore_model_baseline(model: mujoco.MjModel) -> None:
    key = id(model)
    if key not in _MODEL_BASELINES:
        _MODEL_BASELINES[key] = (
            model.geom_friction.copy(),
            model.body_mass.copy(),
            model.body_ipos.copy(),
            model.dof_damping.copy(),
        )
    gf, bm, bip, dd = _MODEL_BASELINES[key]
    model.geom_friction[:] = gf
    model.body_mass[:] = bm
    model.body_ipos[:] = bip
    model.dof_damping[:] = dd


def apply_scenario(model: mujoco.MjModel, scenario: dict[str, Any]) -> None:
    _restore_model_baseline(model)
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        mu = float(scenario.get("floor_friction", 1.0))
        model.geom_friction[floor_id, 0] = mu

    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    if base_id >= 0:
        base_mass = float(scenario.get("base_mass", model.body_mass[base_id]))
        model.body_mass[base_id] = base_mass + float(scenario.get("mass_offset", 0.0))
        lateral = float(scenario.get("com_lateral_offset", 0.0))
        if lateral != 0.0:
            model.body_ipos[base_id, 0] = lateral

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TILT_JOINT)
    if jid >= 0:
        adr = int(model.jnt_dofadr[jid])
        base_damp = float(scenario.get("base_tilt_damping", model.dof_damping[adr]))
        model.dof_damping[adr] = base_damp * float(scenario.get("damping_scale", 1.0))


def reset_state(model: mujoco.MjModel, data: mujoco.MjData, scenario: dict[str, Any]) -> None:
    mujoco.mj_resetData(model, data)
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TILT_JOINT)
    if jid >= 0:
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        data.qpos[qadr] = float(scenario.get("initial_tilt", 0.0))
        data.qvel[dadr] = float(scenario.get("initial_tilt_vel", 0.0))
    mujoco.mj_forward(model, data)


def _sensor_scalar(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> float:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        return 0.0
    adr = int(model.sensor_adr[sid])
    return float(data.sensordata[adr])


def upright_z(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    sl = _sensor_slice(model, "upright_axis")
    if sl is not None:
        axis = np.asarray(data.sensordata[sl], dtype=float)
        if axis.size >= 3:
            return float(axis[2])
    crown_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CROWN_BODY)
    if crown_id < 0:
        return 0.0
    mat = np.asarray(data.xmat[crown_id], dtype=float).reshape(3, 3)
    return float(mat[2, 2])


def _sensor_slice(model: mujoco.MjModel, name: str) -> slice | None:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        return None
    adr = int(model.sensor_adr[sid])
    dim = int(model.sensor_dim[sid])
    return slice(adr, adr + dim)


def tilt_state(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, float]:
    angle = _sensor_scalar(model, data, "tilt_pos")
    rate = _sensor_scalar(model, data, "tilt_vel")
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TILT_JOINT)
    if jid >= 0:
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        angle = float(data.qpos[qadr])
        rate = float(data.qvel[dadr])
    return angle, rate


def observation(
    model: mujoco.MjModel, data: mujoco.MjData, scenario: dict[str, Any], time: float
) -> dict[str, Any]:
    tilt_angle, tilt_vel = tilt_state(model, data)
    return {
        "time": float(time),
        "duration": float(scenario.get("duration", DEFAULT_DURATION)),
        "tilt_angle": float(tilt_angle),
        "tilt_vel": float(tilt_vel),
        "upright_z": float(upright_z(model, data)),
        "floor_friction": float(scenario.get("floor_friction", 1.0)),
        "mass_offset": float(scenario.get("mass_offset", 0.0)),
        "damping_scale": float(scenario.get("damping_scale", 1.0)),
        "com_lateral_offset": float(scenario.get("com_lateral_offset", 0.0)),
    }


def subtree_bodies(model: mujoco.MjModel, root: int) -> set[int]:
    bodies = {root}
    for bid in range(1, model.nbody):
        parent = int(model.body_parentid[bid])
        while parent > 0:
            if parent in bodies:
                bodies.add(bid)
                break
            parent = int(model.body_parentid[parent])
    return bodies


def run_rollout(
    model: mujoco.MjModel,
    policy_fn: Callable[[dict[str, Any]], Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    apply_scenario(model, scenario)
    data = mujoco.MjData(model)
    reset_state(model, data, scenario)

    duration = float(scenario.get("duration", DEFAULT_DURATION))
    dt = float(model.opt.timestep)
    steps = max(1, int(round(duration / dt)))
    hold_steps = max(1, int(round(2.5 / dt)))

    ctrl_lo = float(model.actuator_ctrlrange[0, 0]) if model.nu else -1.0
    ctrl_hi = float(model.actuator_ctrlrange[0, 1]) if model.nu else 1.0

    ctrl_history: list[float] = []
    upright_trace: list[float] = []
    tilt_hold: list[float] = []
    vel_hold: list[float] = []
    recovered = False
    min_upright = 1.0

    for step in range(steps):
        t = step * dt
        obs = observation(model, data, scenario, t)
        action = policy_fn(obs)
        torque = float(np.asarray(action, dtype=float).reshape(-1)[0])
        if not math.isfinite(torque):
            return {"finite": False}
        if model.nu:
            data.ctrl[0] = max(ctrl_lo, min(ctrl_hi, torque))
        mujoco.mj_step(model, data)
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            return {"finite": False}

        uz = upright_z(model, data)
        min_upright = min(min_upright, uz)
        upright_trace.append(uz)
        if uz >= float(scenario.get("recovery_upright_min", 0.82)):
            recovered = True

        tilt_angle, tilt_vel = tilt_state(model, data)
        if step >= steps - hold_steps:
            tilt_hold.append(abs(tilt_angle))
            vel_hold.append(abs(tilt_vel))
        ctrl_history.append(float(data.ctrl[0]) if model.nu else 0.0)

    ctrl_arr = np.asarray(ctrl_history, dtype=float)
    effort = float(np.mean(np.abs(ctrl_arr))) if ctrl_arr.size else 0.0
    jerk = float(np.mean(np.abs(np.diff(ctrl_arr, n=2)))) if ctrl_arr.size >= 3 else 0.0

    return {
        "finite": True,
        "recovered": recovered,
        "min_upright_z": float(min_upright),
        "hold_upright_z": float(np.mean(upright_trace[-hold_steps:])) if upright_trace else 0.0,
        "hold_tilt_abs": float(np.mean(tilt_hold)) if tilt_hold else float("inf"),
        "hold_tilt_vel": float(np.max(vel_hold)) if vel_hold else float("inf"),
        "effort": effort,
        "jerk": jerk,
    }
