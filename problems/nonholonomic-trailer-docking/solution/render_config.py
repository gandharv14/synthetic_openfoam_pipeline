from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mujoco

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from trailer_env import kinematic_step, observation as trailer_observation, reset_data  # noqa: E402


RENDER_SCENARIO: dict[str, Any] = {
    "id": "review_successful_reverse_dock",
    "family": "straight_reverse_dock",
    "duration": 5.8,
    "trailer_length": 0.9,
    "initial_hitch": [1.02, 0.08],
    "initial_tractor_yaw": 0.10,
    "initial_trailer_yaw": 0.04,
    "target_pose": [-0.60, 0.00, 0.00],
    "reverse_required": True,
    "no_go": [
        {"type": "circle", "center": [0.18, -0.34], "radius": 0.12},
    ],
}

_LOGICAL_QVEL = None
_PHYSICS_FREEZE_QVEL = None


def _observation_with_logical_qvel(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    time_sec: float,
) -> dict[str, Any]:
    if _LOGICAL_QVEL is None:
        return trailer_observation(model, data, RENDER_SCENARIO, time_sec)

    physical_qvel = data.qvel.copy()
    try:
        data.qvel[:] = _LOGICAL_QVEL
        return trailer_observation(model, data, RENDER_SCENARIO, time_sec)
    finally:
        data.qvel[:] = physical_qvel


def _freeze_physics_velocity_for_renderer_step(data: mujoco.MjData) -> None:
    # render_mujoco calls mj_step after this hook. The logical velocity is cached
    # above for policy observations; this frozen physical velocity prevents a
    # second integration of the manually advanced kinematic state.
    if _PHYSICS_FREEZE_QVEL is not None:
        data.qvel[:] = _PHYSICS_FREEZE_QVEL


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    global _LOGICAL_QVEL, _PHYSICS_FREEZE_QVEL
    initialized = reset_data(model, RENDER_SCENARIO)
    data.qpos[:] = initialized.qpos
    data.qvel[:] = initialized.qvel
    _LOGICAL_QVEL = data.qvel.copy()
    _PHYSICS_FREEZE_QVEL = data.qvel.copy()
    data.time = 0.0
    mujoco.mj_forward(model, data)


def observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_obs: dict[str, Any],
) -> dict[str, Any]:
    _ = base_obs
    return _observation_with_logical_qvel(model, data, float(data.time))


def before_step(model: mujoco.MjModel, data: mujoco.MjData, policy: Any) -> None:
    global _LOGICAL_QVEL
    obs = _observation_with_logical_qvel(model, data, float(data.time))
    action = policy.act(obs)
    kinematic_step(model, data, RENDER_SCENARIO, action, float(data.time), advance_time=False)
    _LOGICAL_QVEL = data.qvel.copy()
    _freeze_physics_velocity_for_renderer_step(data)
    mujoco.mj_forward(model, data)


def update_scene(
    renderer: mujoco.Renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    _ = model
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.08]
    camera.distance = 2.7
    camera.azimuth = 90.0
    camera.elevation = -55.0
    renderer.update_scene(data, camera=camera)
