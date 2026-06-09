from __future__ import annotations

import numpy as np
import mujoco


INITIAL_QPOS = np.array([0.0, 0.0, 0.04, 0.08, -0.16, 0.08, 0.08, -0.16, 0.08])
PUSH_START = 1.1
PUSH_STOP = 1.22
PUSH_FORCE = 42.0


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_resetData(model, data)
    data.qpos[: INITIAL_QPOS.size] = INITIAL_QPOS
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def before_step(model: mujoco.MjModel, data: mujoco.MjData, policy) -> None:
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    data.xfrc_applied[:] = 0.0
    if PUSH_START <= data.time < PUSH_STOP:
        data.xfrc_applied[torso_id, 0] = PUSH_FORCE

    obs = {
        "time": float(data.time),
        "step": int(round(data.time / max(model.opt.timestep, 1e-4))),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "sensordata": data.sensordata.copy(),
        "ctrl": data.ctrl.copy(),
        "nu": int(model.nu),
        "nq": int(model.nq),
        "nv": int(model.nv),
    }
    action = np.asarray(policy.act(obs), dtype=float).reshape(-1)
    if action.size != model.nu:
        raise ValueError(f"policy action size {action.size} does not match model.nu {model.nu}")
    data.ctrl[:] = np.clip(action, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def update_scene(renderer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [float(data.qpos[0]), 0.0, 0.65]
    camera.distance = 2.5
    camera.azimuth = 70
    camera.elevation = -12
    renderer.update_scene(data, camera=camera)
