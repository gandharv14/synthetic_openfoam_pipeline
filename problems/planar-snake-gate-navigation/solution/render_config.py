from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from snake_env import (  # noqa: E402
    LINK_RADIUS,
    apply_action,
    apply_disturbance,
    body_points,
    build_model,
    gate_passed,
    head_xy,
    indices,
    observation,
    reset_data,
)

RENDER_SCENARIO: dict[str, Any] = {
    "id": "review_clear_s_curve",
    "family": "review",
    "initial_pose": [-0.45, 0.02, 0.02],
    "target": [0.88, 0.02],
    "gates": [
        {"center": [-0.20, 0.02], "yaw": 0.00, "width": 0.46, "depth": 0.22},
        {"center": [0.07, 0.15], "yaw": 0.25, "width": 0.46, "depth": 0.22},
        {"center": [0.35, -0.10], "yaw": -0.28, "width": 0.46, "depth": 0.22},
        {"center": [0.63, 0.09], "yaw": 0.22, "width": 0.46, "depth": 0.22},
        {"center": [0.85, 0.02], "yaw": 0.00, "width": 0.46, "depth": 0.22},
    ],
    "no_go": [
        {"type": "circle", "center": [-0.12, -0.62], "radius": 0.05},
        {"type": "circle", "center": [0.30, 0.62], "radius": 0.05},
        {"type": "circle", "center": [0.63, -0.62], "radius": 0.05},
    ],
    "duration": 10.0,
    "body_friction": 1.05,
    "joint_damping": 0.55,
    "root_damping": 10.5,
    "drive_scale": 0.45,
    "turn_scale": 0.62,
    "joint_target_limit": 0.22,
}

GATE_RGBA = np.array([0.0, 0.78, 0.24, 0.36], dtype=np.float32)
NO_GO_RGBA = np.array([0.95, 0.05, 0.05, 0.35], dtype=np.float32)
TRACE_RGBA = np.array([0.10, 0.18, 0.95, 0.42], dtype=np.float32)
BODY_POINT_RGBA = np.array([1.0, 0.72, 0.10, 0.35], dtype=np.float32)
MARKER_Z = 0.008


class _RenderState:
    def __init__(self) -> None:
        self.gate_index = 0
        self.idx: dict[str, Any] | None = None
        self.trace: list[np.ndarray] = []


STATE = _RenderState()


def _mat_for_yaw(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64).reshape(-1)


def _add_marker(
    renderer: mujoco.Renderer,
    geom_type: mujoco.mjtGeom,
    size: list[float],
    pos: list[float],
    rgba: np.ndarray,
    mat: np.ndarray | None = None,
) -> None:
    scene = renderer.scene
    if scene.ngeom >= scene.maxgeom:
        return
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        geom_type,
        np.array(size, dtype=np.float64),
        np.array(pos, dtype=np.float64),
        mat if mat is not None else np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    scene.ngeom += 1


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    rendered_model = build_model(RENDER_SCENARIO)
    if rendered_model.nq != model.nq:
        raise RuntimeError("render model shape does not match scenario model")
    reset = reset_data(model, RENDER_SCENARIO)
    data.qpos[:] = reset.qpos
    data.qvel[:] = reset.qvel
    mujoco.mj_forward(model, data)
    STATE.gate_index = 0
    STATE.idx = indices(model)
    STATE.trace = []


def before_step(model: mujoco.MjModel, data: mujoco.MjData, policy: Any) -> None:
    if STATE.idx is None:
        STATE.idx = indices(model)
    hxy = head_xy(model, data, STATE.idx)
    while STATE.gate_index < len(RENDER_SCENARIO["gates"]) and gate_passed(
        hxy, RENDER_SCENARIO["gates"][STATE.gate_index]
    ):
        STATE.gate_index += 1
    obs = observation(model, data, RENDER_SCENARIO, float(data.time), STATE.gate_index, STATE.idx)
    action = policy.act(obs)
    apply_action(model, data, action, RENDER_SCENARIO)
    apply_disturbance(model, data, RENDER_SCENARIO, float(data.time))
    if len(STATE.trace) == 0 or np.linalg.norm(hxy - STATE.trace[-1]) > 0.025:
        STATE.trace.append(hxy.copy())
        STATE.trace = STATE.trace[-80:]


def _add_review_markers(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    _ = model
    for gate in RENDER_SCENARIO["gates"]:
        cx, cy = gate["center"]
        width = float(gate.get("width", 0.28))
        depth = float(gate.get("depth", 0.16))
        yaw = float(gate.get("yaw", 0.0))
        _add_marker(
            renderer,
            mujoco.mjtGeom.mjGEOM_BOX,
            [0.5 * depth, 0.5 * width, 0.004],
            [float(cx), float(cy), MARKER_Z],
            GATE_RGBA,
            _mat_for_yaw(yaw),
        )

    for item in RENDER_SCENARIO["no_go"]:
        cx, cy = item["center"]
        radius = float(item["radius"])
        _add_marker(
            renderer,
            mujoco.mjtGeom.mjGEOM_CYLINDER,
            [radius, 0.004, 0.0],
            [float(cx), float(cy), MARKER_Z],
            NO_GO_RGBA,
        )

    for point in STATE.trace[::2]:
        _add_marker(
            renderer,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            [0.014, 0.014, 0.014],
            [float(point[0]), float(point[1]), MARKER_Z + 0.012],
            TRACE_RGBA,
        )

    if STATE.idx is not None:
        for point in body_points(model, data, STATE.idx)[::3]:
            _add_marker(
                renderer,
                mujoco.mjtGeom.mjGEOM_SPHERE,
                [LINK_RADIUS * 0.42, LINK_RADIUS * 0.42, LINK_RADIUS * 0.42],
                [float(point[0]), float(point[1]), MARKER_Z + 0.018],
                BODY_POINT_RGBA,
            )


def update_scene(renderer: mujoco.Renderer, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.0, 0.05]
    camera.distance = 2.55
    camera.azimuth = 90.0
    camera.elevation = -83.0
    renderer.update_scene(data, camera=camera)
    _add_review_markers(renderer, model, data)
