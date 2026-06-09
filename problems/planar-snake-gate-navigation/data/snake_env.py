"""Public MuJoCo helpers for the planar snake gate-navigation task."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

NUM_JOINTS = 6
NUM_LINKS = NUM_JOINTS + 1
ACTION_SIZE = 2 + NUM_JOINTS
LINK_LENGTH = 0.16
LINK_RADIUS = 0.028
DEFAULT_WORKSPACE = {
    "x_min": -2.70,
    "x_max": 1.70,
    "y_min": -1.10,
    "y_max": 1.10,
}


def _model_xml(scenario: dict[str, Any] | None = None) -> str:
    scenario = scenario or {}
    friction = float(scenario.get("body_friction", 0.88))
    joint_damping = float(scenario.get("joint_damping", 0.22))
    root_damping = float(scenario.get("root_damping", 7.5))
    link_mass = float(scenario.get("link_mass", 0.085))

    body_xml = [
        f"""
        <body name="link0" pos="0 0 {LINK_RADIUS + 0.006}">
          <joint name="root_x" type="slide" axis="1 0 0" damping="{root_damping:.4f}" armature="0.02"/>
          <joint name="root_y" type="slide" axis="0 1 0" damping="{root_damping:.4f}" armature="0.02"/>
          <joint name="root_yaw" type="hinge" axis="0 0 1" damping="0.45" armature="0.02"/>
          <geom name="link0_geom" type="capsule" fromto="{-0.5 * LINK_LENGTH:.5f} 0 0 {0.5 * LINK_LENGTH:.5f} 0 0"
                size="{LINK_RADIUS:.5f}" mass="{link_mass:.5f}" friction="{friction:.4f} 0.08 0.02"
                rgba="0.12 0.36 0.68 1"/>
        """
    ]
    for idx in range(1, NUM_LINKS):
        joint_id = idx - 1
        color = "0.16 0.50 0.42 1" if idx % 2 else "0.12 0.36 0.68 1"
        joint_offset = -0.5 * LINK_LENGTH if idx == 1 else -LINK_LENGTH
        body_xml.append(
            f"""
          <body name="link{idx}" pos="{joint_offset:.5f} 0 0">
            <joint name="joint{joint_id}" type="hinge" axis="0 0 1" limited="true"
                   range="-1.15 1.15" damping="{joint_damping:.4f}" armature="0.01"/>
            <geom name="link{idx}_geom" type="capsule" fromto="0 0 0 {-LINK_LENGTH:.5f} 0 0"
                  size="{LINK_RADIUS:.5f}" mass="{link_mass:.5f}" friction="{friction:.4f} 0.08 0.02"
                  rgba="{color}"/>
            """
        )
    body_xml.append("          " + "</body>\n" * NUM_JOINTS + "        </body>")

    actuator_xml = [
        '<motor name="root_x_force" joint="root_x" gear="8" ctrllimited="true" ctrlrange="-1 1"/>',
        '<motor name="root_y_force" joint="root_y" gear="8" ctrllimited="true" ctrlrange="-1 1"/>',
        '<motor name="root_yaw_torque" joint="root_yaw" gear="5" ctrllimited="true" ctrlrange="-1 1"/>',
    ]
    for idx in range(NUM_JOINTS):
        actuator_xml.append(
            f'<position name="joint{idx}_target" joint="joint{idx}" kp="9.5" '
            'ctrllimited="true" ctrlrange="-1.0 1.0"/>'
        )

    return f"""
<mujoco model="planar_snake_gate_navigation">
  <compiler angle="radian" coordinate="local"/>
  <option timestep="0.02" integrator="RK4" iterations="30" cone="elliptic"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <default>
    <joint limited="false"/>
    <geom condim="3" solref="0.02 1" solimp="0.85 0.95 0.001"/>
  </default>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.87 0.88 0.86" rgb2="0.78 0.80 0.78"
             width="512" height="512"/>
    <material name="floor_mat" texture="grid" texrepeat="5 4" reflectance="0.05"/>
  </asset>
  <worldbody>
    <light pos="0 0 2.2" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="floor" type="plane" size="1.5 1.0 0.05" material="floor_mat" friction="1.0 0.08 0.02"/>
    {''.join(body_xml)}
  </worldbody>
  <actuator>
    {' '.join(actuator_xml)}
  </actuator>
</mujoco>
"""


def build_model(scenario: dict[str, Any] | None = None) -> mujoco.MjModel:
    """Build the public articulated snake model.

    Scenario dictionaries may adjust mass, damping, and friction. Hidden gate
    layouts are not encoded in the MJCF; they are supplied through observations
    and scorer-side metrics.
    """

    return mujoco.MjModel.from_xml_string(_model_xml(scenario))


def indices(model: mujoco.MjModel) -> dict[str, Any]:
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"link{i}") for i in range(NUM_LINKS)]
    geom_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"link{i}_geom") for i in range(NUM_LINKS)]
    return {
        "body_ids": body_ids,
        "geom_ids": geom_ids,
        "root_qpos": [0, 1, 2],
        "joint_qpos": list(range(3, 3 + NUM_JOINTS)),
        "root_qvel": [0, 1, 2],
        "joint_qvel": list(range(3, 3 + NUM_JOINTS)),
    }


def reset_data(model: mujoco.MjModel, scenario: dict[str, Any]) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    pose = scenario.get("initial_pose", [-0.86, 0.0, 0.0])
    data.qpos[0] = float(pose[0])
    data.qpos[1] = float(pose[1])
    data.qpos[2] = float(pose[2])
    phase = float(scenario.get("initial_joint_phase", 0.0))
    for idx in range(NUM_JOINTS):
        data.qpos[3 + idx] = 0.18 * math.sin(phase - 0.72 * idx)
    mujoco.mj_forward(model, data)
    return data


def head_xy(model: mujoco.MjModel, data: mujoco.MjData, idx: dict[str, Any] | None = None) -> np.ndarray:
    _ = model, idx
    return np.array([float(data.qpos[0]), float(data.qpos[1])], dtype=float)


def head_yaw(model: mujoco.MjModel, data: mujoco.MjData, idx: dict[str, Any] | None = None) -> float:
    _ = model, idx
    return wrap_angle(float(data.qpos[2]))


def body_points(model: mujoco.MjModel, data: mujoco.MjData, idx: dict[str, Any] | None = None) -> list[np.ndarray]:
    idx = idx or indices(model)
    points: list[np.ndarray] = []
    for link_idx, body_id in enumerate(idx["body_ids"]):
        pos = np.array(data.xpos[body_id][:2], dtype=float)
        axis = np.array(data.xmat[body_id].reshape(3, 3)[:2, 0], dtype=float)
        if link_idx == 0:
            points.extend([pos - 0.5 * LINK_LENGTH * axis, pos, pos + 0.5 * LINK_LENGTH * axis])
        else:
            points.extend([pos, pos - 0.5 * LINK_LENGTH * axis, pos - LINK_LENGTH * axis])
    return points


def tail_xy(model: mujoco.MjModel, data: mujoco.MjData, idx: dict[str, Any] | None = None) -> np.ndarray:
    idx = idx or indices(model)
    body_id = idx["body_ids"][-1]
    pos = np.array(data.xpos[body_id][:2], dtype=float)
    axis = np.array(data.xmat[body_id].reshape(3, 3)[:2, 0], dtype=float)
    return pos - LINK_LENGTH * axis


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def gate_local_error(point: np.ndarray, gate: dict[str, Any]) -> tuple[float, float, float]:
    center = np.array(gate["center"], dtype=float)
    yaw = float(gate.get("yaw", 0.0))
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
    lateral_axis = np.array([-math.sin(yaw), math.cos(yaw)], dtype=float)
    delta = np.array(point, dtype=float) - center
    longitudinal = float(np.dot(delta, forward))
    lateral = float(np.dot(delta, lateral_axis))
    distance = float(np.linalg.norm(delta))
    return longitudinal, lateral, distance


def gate_passed(point: np.ndarray, gate: dict[str, Any]) -> bool:
    longitudinal, lateral, distance = gate_local_error(point, gate)
    half_width = 0.5 * float(gate.get("width", 0.28))
    depth = float(gate.get("depth", 0.18))
    capture = float(gate.get("capture_radius", max(0.12, half_width)))
    return (abs(lateral) <= half_width and -depth <= longitudinal <= depth) or distance <= capture


def clip_action(action: Any, scenario: dict[str, Any] | None = None) -> np.ndarray:
    scenario = scenario or {}
    values = np.asarray(action, dtype=float).reshape(-1)
    if values.size == 0:
        values = np.zeros(ACTION_SIZE, dtype=float)
    if values.size < ACTION_SIZE:
        padded = np.zeros(ACTION_SIZE, dtype=float)
        padded[: values.size] = values
        values = padded
    values = values[:ACTION_SIZE]
    if not np.isfinite(values).all():
        raise ValueError("action contains non-finite values")
    values[0] = np.clip(values[0], -1.0, 1.0)
    values[1] = np.clip(values[1], -1.0, 1.0)
    joint_limit = float(scenario.get("joint_target_limit", 1.0))
    values[2:] = np.clip(values[2:], -joint_limit, joint_limit)
    return values


def apply_action(model: mujoco.MjModel, data: mujoco.MjData, action: Any, scenario: dict[str, Any] | None = None) -> np.ndarray:
    scenario = scenario or {}
    values = clip_action(action, scenario)
    yaw = head_yaw(model, data)
    drive_scale = float(scenario.get("drive_scale", 0.46))
    turn_scale = float(scenario.get("turn_scale", 0.64))
    drive = drive_scale * float(values[0])
    data.ctrl[0] = drive * math.cos(yaw)
    data.ctrl[1] = drive * math.sin(yaw)
    data.ctrl[2] = turn_scale * float(values[1])
    data.ctrl[3 : 3 + NUM_JOINTS] = values[2:]
    return values


def apply_disturbance(model: mujoco.MjModel, data: mujoco.MjData, scenario: dict[str, Any], time_sec: float) -> None:
    _ = model
    data.qfrc_applied[:] = 0.0
    for event in scenario.get("disturbances", []):
        start = float(event.get("start", 0.0))
        duration = float(event.get("duration", 0.0))
        if start <= time_sec <= start + duration:
            force = np.asarray(event.get("force", [0.0, 0.0]), dtype=float)
            data.qfrc_applied[0] += float(force[0])
            data.qfrc_applied[1] += float(force[1])


def active_gate(scenario: dict[str, Any], gate_index: int) -> dict[str, Any]:
    gates = scenario.get("gates", [])
    if not gates:
        return {"center": scenario.get("target", [0.0, 0.0]), "yaw": 0.0, "width": 0.30}
    return gates[min(max(gate_index, 0), len(gates) - 1)]


def observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: dict[str, Any],
    time_sec: float,
    gate_index: int,
    idx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    idx = idx or indices(model)
    gate = active_gate(scenario, gate_index)
    next_gate = None
    gates = scenario.get("gates", [])
    if gate_index + 1 < len(gates):
        next_gate = gates[gate_index + 1]
    hxy = head_xy(model, data, idx)
    yaw = head_yaw(model, data, idx)
    velocity_world = np.array([float(data.qvel[0]), float(data.qvel[1])], dtype=float)
    forward = np.array([math.cos(yaw), math.sin(yaw)], dtype=float)
    lateral = np.array([-math.sin(yaw), math.cos(yaw)], dtype=float)
    body_xy = body_points(model, data, idx)
    return {
        "time": float(time_sec),
        "action_size": ACTION_SIZE,
        "num_joints": NUM_JOINTS,
        "head_xy": hxy.tolist(),
        "head_yaw": yaw,
        "head_velocity_world": velocity_world.tolist(),
        "head_velocity_body": [float(np.dot(velocity_world, forward)), float(np.dot(velocity_world, lateral))],
        "tail_xy": tail_xy(model, data, idx).tolist(),
        "joint_angles": np.asarray(data.qpos[idx["joint_qpos"]], dtype=float).tolist(),
        "joint_velocities": np.asarray(data.qvel[idx["joint_qvel"]], dtype=float).tolist(),
        "body_points": [point.tolist() for point in body_xy],
        "gate_index": int(gate_index),
        "num_gates": len(gates),
        "target_gate": gate,
        "next_gate": next_gate,
        "final_target": scenario.get("target", gates[-1]["center"] if gates else [0.0, 0.0]),
        "no_go": scenario.get("no_go", []),
        "workspace": scenario.get("workspace", DEFAULT_WORKSPACE),
        "drive_scale": float(scenario.get("drive_scale", 0.46)),
        "turn_scale": float(scenario.get("turn_scale", 0.64)),
    }


def workspace_margin(point: np.ndarray, workspace: dict[str, float] | None = None, radius: float = LINK_RADIUS) -> float:
    workspace = workspace or DEFAULT_WORKSPACE
    return min(
        float(point[0]) - float(workspace["x_min"]) - radius,
        float(workspace["x_max"]) - float(point[0]) - radius,
        float(point[1]) - float(workspace["y_min"]) - radius,
        float(workspace["y_max"]) - float(point[1]) - radius,
    )


def no_go_clearance(point: np.ndarray, no_go: list[dict[str, Any]], radius: float = LINK_RADIUS) -> float:
    clearances: list[float] = []
    for item in no_go:
        if item.get("type") != "circle":
            continue
        center = np.array(item["center"], dtype=float)
        clearances.append(float(np.linalg.norm(np.array(point, dtype=float) - center) - float(item["radius"]) - radius))
    return min(clearances) if clearances else 1.0
