from __future__ import annotations

import math

import mujoco
import numpy as np

CASE = {
    "base": np.array([0.05, 0.14, 0.20, 0.08, 0.18, 0.09, 0.04], dtype=float),
    "amplitude": np.array([0.18, 0.18, 0.15, 0.11, 0.15, 0.17, 0.12], dtype=float),
    "frequency": 0.16,
    "phase": np.array([0.7, 1.3, 2.2, 3.1, 0.4, 1.6, 2.6], dtype=float),
    "actuator_gains": np.array([0.74, 0.76, 0.66, 0.58, 0.72, 0.64, 0.56], dtype=float),
    "dropouts": [
        {"joint": 1, "start": 2.05, "duration": 0.18, "gain": 0.16},
        {"joint": 6, "start": 4.80, "duration": 0.18, "gain": 0.12},
    ],
    "gusts": [
        {"time": 2.95, "joint": 2, "impulse": -0.24, "duration": 0.05},
        {"time": 5.40, "joint": 0, "impulse": 0.29, "duration": 0.05},
    ],
}

LEFT_TIP_SITE = "left_tip_site"
RIGHT_TIP_SITE = "right_tip_site"
CONTROL_SKIP = 2
_LAST_CTRL: np.ndarray | None = None


def _target(t: float) -> tuple[np.ndarray, np.ndarray]:
    omega = 2.0 * math.pi * float(CASE["frequency"])
    arg = omega * t + CASE["phase"]
    q = CASE["base"] + CASE["amplitude"] * np.sin(arg)
    qd = CASE["amplitude"] * omega * np.cos(arg)
    return q, qd


def _tip_positions(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = mujoco.MjData(model)
    d.qpos[:] = qpos
    d.qvel[:] = 0.0
    mujoco.mj_forward(model, d)
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_TIP_SITE)
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_TIP_SITE)
    return d.site_xpos[left].copy(), d.site_xpos[right].copy()


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    global _LAST_CTRL
    mujoco.mj_resetData(model, data)
    q0, _ = _target(0.0)
    data.qpos[:] = q0
    data.qvel[:] = 0.0
    _LAST_CTRL = np.zeros(model.nu)
    mujoco.mj_forward(model, data)


def _dynamic_gain(t: float, nu: int) -> np.ndarray:
    gains = np.asarray(CASE["actuator_gains"], dtype=float).copy()
    for dropout in CASE["dropouts"]:
        if float(dropout["start"]) <= t < float(dropout["start"]) + float(dropout["duration"]):
            gains[int(dropout["joint"])] *= float(dropout["gain"])
    return gains[:nu]


def _apply_gusts(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    data.qfrc_applied[:] = 0.0
    for gust in CASE["gusts"]:
        start = float(gust["time"])
        duration = float(gust.get("duration", 0.05))
        if start <= data.time < start + duration:
            data.qfrc_applied[int(gust["joint"])] += float(gust["impulse"]) / max(
                duration, model.opt.timestep
            )


def before_step(model: mujoco.MjModel, data: mujoco.MjData, policy) -> None:
    global _LAST_CTRL
    if _LAST_CTRL is None or _LAST_CTRL.size != model.nu:
        _LAST_CTRL = np.zeros(model.nu)
    step = int(round(data.time / max(model.opt.timestep, 1e-4)))
    q_ref, _ = _target(float(data.time))
    target_left, target_right = _tip_positions(model, q_ref)
    left = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_TIP_SITE)
    right = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_TIP_SITE)
    obs = {
        "time": float(data.time),
        "step": step,
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "left_tip_pos": data.site_xpos[left].copy(),
        "right_tip_pos": data.site_xpos[right].copy(),
        "target_left_tip_pos": target_left,
        "target_right_tip_pos": target_right,
        "target_root_roll": float(q_ref[0]),
        "target_left_camber": float(np.mean(q_ref[1:4])),
        "target_right_camber": float(np.mean(q_ref[4:7])),
        "last_ctrl": _LAST_CTRL.copy(),
        "phase": float((float(data.time) * float(CASE["frequency"])) % 1.0),
    }
    if step % CONTROL_SKIP == 0:
        action = np.asarray(policy.act(obs), dtype=float).reshape(-1)
        if action.size != model.nu:
            raise ValueError(f"policy action size {action.size} does not match model.nu {model.nu}")
        _LAST_CTRL = np.clip(action, -1.0, 1.0)
    _apply_gusts(model, data)
    data.ctrl[:] = np.clip(_LAST_CTRL * _dynamic_gain(float(data.time), model.nu), -1.0, 1.0)


def update_scene(renderer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.05, 0.0, 1.18]
    camera.distance = 2.05
    camera.azimuth = 125
    camera.elevation = -17
    renderer.update_scene(data, camera=camera)

    q_ref, _ = _target(float(data.time))
    targets = _tip_positions(model, q_ref)
    scene = renderer.scene
    colors = [
        np.array([0.15, 0.95, 0.25, 0.78], dtype=float),
        np.array([0.95, 0.85, 0.20, 0.78], dtype=float),
    ]
    for target, color in zip(targets, colors, strict=True):
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mat = np.eye(3, dtype=float).reshape(-1)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([0.026, 0.0, 0.0], dtype=float),
            target,
            mat,
            color,
        )
        scene.ngeom += 1
