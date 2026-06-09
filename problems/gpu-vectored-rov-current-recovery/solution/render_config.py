from __future__ import annotations

import math

import mujoco
import numpy as np

CAMERA_OFFSET = 0.43
BASE_DAMPING = np.array([4.2, 4.4, 5.0, 1.25, 1.35, 1.15], dtype=float)
CONTROL_SKIP = 2
LAST_CTRL = np.zeros(8, dtype=float)

CASE = {
    "id": "review-rov-current",
    "duration": 7.0,
    "frequency": 0.085,
    "target_base": np.array([0.08, -0.05, 0.90], dtype=float),
    "target_amplitude": np.array([0.15, 0.13, 0.06], dtype=float),
    "phase": np.array([0.4, 1.3, 2.2, 0.8], dtype=float),
    "yaw_base": 0.05,
    "yaw_amplitude": 0.42,
    "drag_scale": 1.05,
    "current_bias": np.array([0.34, -0.24, 0.08, 0.03, -0.03, 0.04], dtype=float),
    "current_amplitude": np.array([0.62, 0.48, 0.28, 0.08, 0.07, 0.12], dtype=float),
    "actuator_gains": np.array([0.96, 0.94, 0.98, 0.92, 1.0, 0.96, 0.98, 0.94], dtype=float),
    "dropouts": [{"thruster": 3, "start": 2.60, "duration": 0.36, "gain": 0.18}],
    "impulses": [{"time": 4.55, "duration": 0.08, "wrench": [1.7, -1.0, 0.6, 0.18, -0.12, 0.26]}],
    "initial_position": np.array([-0.22, 0.18, 0.82], dtype=float),
    "initial_yaw": -0.26,
}


def _quat_from_yaw(yaw: float) -> np.ndarray:
    return np.array([math.cos(0.5 * yaw), 0.0, 0.0, math.sin(0.5 * yaw)], dtype=float)


def _target(t: float) -> dict[str, np.ndarray | float]:
    omega = 2.0 * math.pi * float(CASE["frequency"])
    camera = CASE["target_base"] + CASE["target_amplitude"] * np.sin(omega * t + CASE["phase"][:3])
    yaw = float(CASE["yaw_base"] + float(CASE["yaw_amplitude"]) * math.sin(omega * t + float(CASE["phase"][3])))
    heading = np.array([math.cos(yaw), math.sin(yaw), 0.0], dtype=float)
    position = camera - CAMERA_OFFSET * heading
    return {"position": position, "camera": camera, "heading": heading, "yaw": yaw}


def _dynamic_gain(t: float, nu: int) -> np.ndarray:
    gains = np.asarray(CASE["actuator_gains"], dtype=float).copy()
    for dropout in CASE["dropouts"]:
        start = float(dropout["start"])
        if start <= t < start + float(dropout["duration"]):
            gains[int(dropout["thruster"])] *= float(dropout["gain"])
    return gains[:nu]


def _disturbance(t: float) -> np.ndarray:
    omega = 2.0 * math.pi * float(CASE["frequency"]) * 1.7
    current = CASE["current_bias"] + CASE["current_amplitude"] * np.sin(omega * t + CASE["phase"][0] + np.arange(6) * 0.61)
    for impulse in CASE["impulses"]:
        start = float(impulse["time"])
        duration = float(impulse["duration"])
        if start <= t < start + duration:
            current += np.asarray(impulse["wrench"], dtype=float) / duration
    return current


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    global LAST_CTRL
    model.dof_damping[:] = BASE_DAMPING * float(CASE["drag_scale"])
    mujoco.mj_resetData(model, data)
    data.qpos[:3] = CASE["initial_position"]
    data.qpos[3:7] = _quat_from_yaw(float(CASE["initial_yaw"]))
    data.qvel[:] = 0.0
    LAST_CTRL = np.zeros(model.nu, dtype=float)
    mujoco.mj_forward(model, data)


def before_step(model: mujoco.MjModel, data: mujoco.MjData, policy) -> None:
    global LAST_CTRL
    step = int(round(data.time / max(model.opt.timestep, 1.0e-4)))
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rov")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "camera_site")
    rot = data.xmat[body_id].reshape(3, 3).copy()
    target = _target(float(data.time))
    if step % CONTROL_SKIP == 0:
        obs = {
            "time": float(data.time),
            "step": step,
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "position": data.xpos[body_id].copy(),
            "rotation_matrix": rot,
            "heading": rot[:, 0].copy(),
            "up_axis": rot[:, 2].copy(),
            "camera_pos": data.site_xpos[site_id].copy(),
            "target_position": np.asarray(target["position"], dtype=float),
            "target_camera_pos": np.asarray(target["camera"], dtype=float),
            "target_heading": np.asarray(target["heading"], dtype=float),
            "target_yaw": float(target["yaw"]),
            "last_ctrl": LAST_CTRL.copy(),
            "actuator_gear": model.actuator_gear[:, : model.nv].copy(),
            "phase": float((float(data.time) * float(CASE["frequency"])) % 1.0),
        }
        action = np.asarray(policy.act(obs), dtype=float).reshape(-1)
        if action.size != model.nu:
            raise ValueError(f"policy action size {action.size} does not match model.nu {model.nu}")
        LAST_CTRL = np.clip(action, -1.0, 1.0)
    data.qfrc_applied[:] = _disturbance(float(data.time))
    data.ctrl[:] = np.clip(
        LAST_CTRL * _dynamic_gain(float(data.time), model.nu),
        -1.0,
        1.0,
    )


def update_scene(renderer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.02, 0.0, 0.86]
    camera.distance = 1.95
    camera.azimuth = 126
    camera.elevation = -18
    renderer.update_scene(data, camera=camera)

    target = _target(float(data.time))
    scene = renderer.scene
    markers = [
        (np.asarray(target["camera"], dtype=float), np.array([1.0, 0.72, 0.16, 0.82], dtype=float), 0.032),
        (np.asarray(target["position"], dtype=float), np.array([0.20, 0.95, 0.40, 0.55], dtype=float), 0.022),
    ]
    for pos, color, radius in markers:
        if scene.ngeom >= scene.maxgeom:
            break
        geom = scene.geoms[scene.ngeom]
        mat = np.eye(3, dtype=float).reshape(-1)
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            np.array([radius, 0.0, 0.0], dtype=float),
            pos,
            mat,
            color,
        )
        scene.ngeom += 1
