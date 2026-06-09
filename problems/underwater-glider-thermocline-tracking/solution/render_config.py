from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from glider_env import build_model, kinematic_step, observation, reset_data, sample_reached  # noqa: E402


RENDER_SCENARIO: dict[str, Any] = {
    "id": "review_underwater_glider_internal_wave",
    "duration": 10.5,
    "start": [0.00, 0.74],
    "finish": [2.00, 0.84],
    "workspace": {"x_min": -0.10, "x_max": 2.14, "z_min": 0.20, "z_max": 1.40},
    "thermocline": {
        "base": 0.74,
        "slope": 0.045,
        "waves": [
            {"amp": 0.125, "freq": 3.35, "phase": 0.95},
            {"amp": -0.046, "freq": 7.40, "phase": 0.35},
        ],
    },
    "samples": [
        {"x": 0.38, "radius_x": 0.055, "radius_z": 0.060},
        {"x": 0.78, "radius_x": 0.052, "radius_z": 0.058},
        {"x": 1.18, "radius_x": 0.052, "radius_z": 0.058},
        {"x": 1.58, "radius_x": 0.055, "radius_z": 0.060},
    ],
    "plumes": [
        {"center": [0.58, 1.05], "radius": 0.110},
        {"center": [1.14, 0.50], "radius": 0.105},
        {"center": [1.62, 1.02], "radius": 0.105},
    ],
    "base_current": [0.015, -0.009],
    "shear": {"amplitude": 0.042, "frequency": 5.25, "phase": 0.70},
    "eddies": [
        {"center": [0.90, 0.86], "strength": 0.007},
        {"center": [1.44, 0.74], "strength": -0.006},
    ],
}

TRACE_RGBA = np.array([1.0, 0.84, 0.12, 0.48], dtype=np.float32)
ACTIVE_SAMPLE_RGBA = np.array([0.05, 1.0, 0.38, 0.58], dtype=np.float32)
MARKER_Z = 0.040


class _State:
    def __init__(self) -> None:
        self.sample_index = 0
        self.sample_hold_counter = 0
        self.trace: list[np.ndarray] = []
        self.logical_qvel: np.ndarray | None = None


STATE = _State()


def _add_marker(renderer: mujoco.Renderer, geom_type: mujoco.mjtGeom, size, pos, rgba) -> None:
    scene = renderer.scene
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        geom_type,
        np.array(size, dtype=np.float64),
        np.array(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    reset = reset_data(model, RENDER_SCENARIO)
    data.qpos[:] = reset.qpos
    data.qvel[:] = reset.qvel
    data.time = 0.0
    STATE.sample_index = 0
    STATE.sample_hold_counter = 0
    STATE.trace = []
    STATE.logical_qvel = data.qvel.copy()
    mujoco.mj_forward(model, data)


def before_step(model: mujoco.MjModel, data: mujoco.MjData, policy: Any) -> None:
    if STATE.logical_qvel is not None:
        data.qvel[:] = STATE.logical_qvel
    point = np.array(data.qpos[:2], dtype=float)
    samples = RENDER_SCENARIO["samples"]
    hold_steps = max(
        1,
        int(np.ceil(float(RENDER_SCENARIO.get("sample_hold_time", 0.18)) / float(model.opt.timestep))),
    )
    if STATE.sample_index < len(samples) and sample_reached(point, RENDER_SCENARIO, samples[STATE.sample_index]):
        STATE.sample_hold_counter += 1
        if STATE.sample_hold_counter >= hold_steps:
            STATE.sample_index += 1
            STATE.sample_hold_counter = 0
    elif STATE.sample_index < len(samples):
        STATE.sample_hold_counter = 0
    hold_progress = STATE.sample_hold_counter / hold_steps if STATE.sample_index < len(samples) else 1.0
    obs = observation(model, data, RENDER_SCENARIO, float(data.time), STATE.sample_index, hold_progress)
    action = policy.act(obs)
    kinematic_step(model, data, RENDER_SCENARIO, action, float(data.time), advance_time=False)
    STATE.logical_qvel = data.qvel.copy()
    point = np.array(data.qpos[:2], dtype=float)
    if len(STATE.trace) == 0 or np.linalg.norm(point - STATE.trace[-1]) > 0.030:
        STATE.trace.append(point.copy())
        STATE.trace = STATE.trace[-140:]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def update_scene(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    _ = model
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1.02, 0.80, 0.05]
    camera.distance = 2.68
    camera.azimuth = 90.0
    camera.elevation = -90.0
    renderer.update_scene(data, camera=camera)

    for point in STATE.trace[::2]:
        _add_marker(
            renderer,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.012, 0.012, 0.012],
            [float(point[0]), float(point[1]), MARKER_Z],
            TRACE_RGBA,
        )

    samples = RENDER_SCENARIO["samples"]
    if STATE.sample_index < len(samples):
        sample = samples[STATE.sample_index]
        cx = float(sample["x"])
        from glider_env import target_depth_at  # local import keeps render config compact

        cz = float(sample.get("z", target_depth_at(RENDER_SCENARIO, cx)))
        _add_marker(
            renderer,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [float(sample.get("radius_z", 0.06)), 0.004, 0.0],
            [cx, cz, MARKER_Z + 0.004],
            ACTIVE_SAMPLE_RGBA,
        )
