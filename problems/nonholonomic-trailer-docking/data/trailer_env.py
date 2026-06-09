"""Public deterministic helper for the nonholonomic trailer docking task."""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

DEFAULT_TIMESTEP = 0.02
DEFAULT_WORKSPACE = {
    "x_min": -1.55,
    "x_max": 1.60,
    "y_min": -1.05,
    "y_max": 1.05,
}
DEFAULT_TRAILER_LENGTH = 0.9
DEFAULT_MAX_DRIVE_SPEED = 0.58
DEFAULT_MAX_YAW_RATE = 1.25
TRACTOR_LENGTH = 0.42
TRACTOR_WIDTH = 0.20
TRAILER_WIDTH = 0.22
SAFETY_RADIUS = 0.08


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _unit(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw), math.sin(yaw)], dtype=float)


def _no_go_geoms(no_go: list[dict[str, Any]]) -> str:
    geoms: list[str] = []
    for idx, item in enumerate(no_go):
        if item.get("type") != "circle":
            continue
        cx, cy = item.get("center", [0.0, 0.0])
        radius = float(item.get("radius", 0.1))
        geoms.append(
            f'<geom name="no_go_{idx}" type="cylinder" pos="{float(cx)} {float(cy)} 0.012" '
            f'size="{radius} 0.012" rgba="0.9 0.1 0.1 0.32" contype="0" conaffinity="0"/>'
        )
    return "\n    ".join(geoms)


def _target_geoms(scenario: dict[str, Any], trailer_length: float) -> str:
    tx, ty, tyaw = scenario.get("target_pose", [0.0, 0.0, 0.0])
    footprint_half_length = trailer_length * 0.36
    return f"""
    <body name="target_trailer_pose" pos="{float(tx)} {float(ty)} 0.018" euler="0 0 {float(tyaw)}">
      <geom name="target_trailer_footprint" type="box" pos="0 0 0"
            size="{footprint_half_length} {TRAILER_WIDTH * 0.5} 0.010"
            rgba="0.05 0.75 0.18 0.30" contype="0" conaffinity="0"/>
      <geom name="target_yaw_axis" type="capsule" fromto="-{footprint_half_length} 0 0.035 {footprint_half_length} 0 0.035"
            size="0.012" rgba="0.02 0.55 0.10 0.88" contype="0" conaffinity="0"/>
      <site name="target_center" pos="0 0 0.055" size="0.030" rgba="0.02 0.80 0.10 0.92"/>
    </body>
"""


def build_model(scenario: dict[str, Any]) -> mujoco.MjModel:
    """Build a lightweight MuJoCo geometry model for one trailer scenario."""
    trailer_length = float(scenario.get("trailer_length", DEFAULT_TRAILER_LENGTH))
    workspace = scenario.get("workspace", DEFAULT_WORKSPACE)
    floor_x = 0.5 * (float(workspace["x_max"]) - float(workspace["x_min"]))
    floor_y = 0.5 * (float(workspace["y_max"]) - float(workspace["y_min"]))
    no_go_xml = _no_go_geoms(scenario.get("no_go", []))
    target_xml = _target_geoms(scenario, trailer_length)
    xml = f"""
<mujoco model="nonholonomic_trailer_docking">
  <compiler angle="radian" inertiafromgeom="true"/>
  <option timestep="{float(scenario.get("dt", DEFAULT_TIMESTEP))}" integrator="Euler"
          gravity="0 0 0" iterations="20" tolerance="1e-9"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <worldbody>
    <geom name="workspace" type="plane" size="{floor_x} {floor_y} 0.02"
          rgba="0.82 0.84 0.86 1" contype="0" conaffinity="0"/>
    {no_go_xml}
    {target_xml}
    <body name="tractor" pos="0 0 0.04">
      <joint name="hitch_x" type="slide" axis="1 0 0"/>
      <joint name="hitch_y" type="slide" axis="0 1 0"/>
      <joint name="tractor_yaw" type="hinge" axis="0 0 1"/>
      <geom name="tractor_body" type="box" pos="{TRACTOR_LENGTH * 0.5} 0 0"
            size="{TRACTOR_LENGTH * 0.5} {TRACTOR_WIDTH * 0.5} 0.04"
            rgba="0.12 0.34 0.70 1" contype="0" conaffinity="0"/>
      <site name="hitch_site" pos="0 0 0.055" size="0.025" rgba="0.02 0.02 0.02 1"/>
      <site name="tractor_nose" pos="{TRACTOR_LENGTH} 0 0.055" size="0.02" rgba="0.12 0.34 0.70 1"/>
      <body name="trailer" pos="0 0 0">
        <joint name="trailer_hinge" type="hinge" axis="0 0 1"/>
        <geom name="drawbar" type="capsule" fromto="0 0 0.02 -{trailer_length} 0 0.02"
              size="0.018" rgba="0.08 0.08 0.08 1" contype="0" conaffinity="0"/>
        <geom name="trailer_body" type="box" pos="-{trailer_length * 0.5} 0 0.02"
              size="{trailer_length * 0.36} {TRAILER_WIDTH * 0.5} 0.035"
              rgba="0.92 0.45 0.12 1" contype="0" conaffinity="0"/>
        <site name="trailer_center" pos="-{trailer_length * 0.5} 0 0.055"
              size="0.025" rgba="0.95 0.15 0.05 1"/>
        <site name="trailer_tail" pos="-{trailer_length} 0 0.055"
              size="0.018" rgba="0.95 0.15 0.05 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def indices(model: mujoco.MjModel) -> dict[str, int]:
    """Return qpos/qvel/site/body addresses used by the helper and grader."""
    result: dict[str, int] = {}
    for name in ("hitch_x", "hitch_y", "tractor_yaw", "trailer_hinge"):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        result[f"{name}_qpos"] = int(model.jnt_qposadr[jid])
        result[f"{name}_qvel"] = int(model.jnt_dofadr[jid])
    for name in ("hitch_site", "tractor_nose", "trailer_center", "trailer_tail"):
        result[f"{name}_site"] = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name))
    return result


def reset_data(model: mujoco.MjModel, scenario: dict[str, Any]) -> mujoco.MjData:
    """Create deterministic MjData at the scenario initial pose."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    idx = indices(model)
    hx, hy = scenario.get("initial_hitch", [0.0, 0.0])
    tractor_yaw = float(scenario.get("initial_tractor_yaw", 0.0))
    trailer_yaw = float(scenario.get("initial_trailer_yaw", tractor_yaw))
    data.qpos[idx["hitch_x_qpos"]] = float(hx)
    data.qpos[idx["hitch_y_qpos"]] = float(hy)
    data.qpos[idx["tractor_yaw_qpos"]] = tractor_yaw
    data.qpos[idx["trailer_hinge_qpos"]] = wrap_angle(trailer_yaw - tractor_yaw)
    mujoco.mj_forward(model, data)
    return data


def hitch_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    idx = indices(model)
    return np.array([data.qpos[idx["hitch_x_qpos"]], data.qpos[idx["hitch_y_qpos"]]], dtype=float)


def tractor_yaw(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return wrap_angle(float(data.qpos[indices(model)["tractor_yaw_qpos"]]))


def hitch_angle(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return wrap_angle(float(data.qpos[indices(model)["trailer_hinge_qpos"]]))


def trailer_yaw(model: mujoco.MjModel, data: mujoco.MjData) -> float:
    return wrap_angle(tractor_yaw(model, data) + hitch_angle(model, data))


def trailer_center_xy(model: mujoco.MjModel, data: mujoco.MjData, scenario: dict[str, Any]) -> np.ndarray:
    length = float(scenario.get("trailer_length", DEFAULT_TRAILER_LENGTH))
    return hitch_xy(model, data) - 0.5 * length * _unit(trailer_yaw(model, data))


def trailer_tail_xy(model: mujoco.MjModel, data: mujoco.MjData, scenario: dict[str, Any]) -> np.ndarray:
    length = float(scenario.get("trailer_length", DEFAULT_TRAILER_LENGTH))
    return hitch_xy(model, data) - length * _unit(trailer_yaw(model, data))


def tractor_nose_xy(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    return hitch_xy(model, data) + TRACTOR_LENGTH * _unit(tractor_yaw(model, data))


def desired_hitch_xy(scenario: dict[str, Any]) -> np.ndarray:
    tx, ty, tyaw = scenario["target_pose"]
    length = float(scenario.get("trailer_length", DEFAULT_TRAILER_LENGTH))
    return np.array([float(tx), float(ty)], dtype=float) + 0.5 * length * _unit(float(tyaw))


def _point_velocity(model: mujoco.MjModel, data: mujoco.MjData, scenario: dict[str, Any]) -> tuple[np.ndarray, float, float]:
    idx = indices(model)
    hv = np.array([data.qvel[idx["hitch_x_qvel"]], data.qvel[idx["hitch_y_qvel"]]], dtype=float)
    psi_dot = float(data.qvel[idx["tractor_yaw_qvel"]])
    phi_dot = float(data.qvel[idx["trailer_hinge_qvel"]])
    theta_dot = psi_dot + phi_dot
    length = float(scenario.get("trailer_length", DEFAULT_TRAILER_LENGTH))
    theta = trailer_yaw(model, data)
    center_v = hv - 0.5 * length * theta_dot * np.array([-math.sin(theta), math.cos(theta)], dtype=float)
    return center_v, psi_dot, theta_dot


def observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: dict[str, Any],
    time_sec: float,
) -> dict[str, Any]:
    """Return the public observation dictionary consumed by policies."""
    hxy = hitch_xy(model, data)
    cxy = trailer_center_xy(model, data, scenario)
    tail = trailer_tail_xy(model, data, scenario)
    nose = tractor_nose_xy(model, data)
    target = np.array(scenario["target_pose"], dtype=float)
    desired_hitch = desired_hitch_xy(scenario)
    center_v, psi_dot, theta_dot = _point_velocity(model, data, scenario)
    idx = indices(model)
    return {
        "time": float(time_sec),
        "dt": float(model.opt.timestep),
        "duration": float(scenario.get("duration", 8.0)),
        "remaining_time": max(0.0, float(scenario.get("duration", 8.0)) - float(time_sec)),
        "hitch_x": float(hxy[0]),
        "hitch_y": float(hxy[1]),
        "hitch_vx": float(data.qvel[idx["hitch_x_qvel"]]),
        "hitch_vy": float(data.qvel[idx["hitch_y_qvel"]]),
        "tractor_yaw": tractor_yaw(model, data),
        "tractor_yaw_rate": psi_dot,
        "trailer_x": float(cxy[0]),
        "trailer_y": float(cxy[1]),
        "trailer_vx": float(center_v[0]),
        "trailer_vy": float(center_v[1]),
        "trailer_speed": float(np.linalg.norm(center_v)),
        "trailer_yaw": trailer_yaw(model, data),
        "trailer_yaw_rate": theta_dot,
        "hitch_angle": hitch_angle(model, data),
        "hitch_angle_rate": float(data.qvel[idx["trailer_hinge_qvel"]]),
        "target_x": float(target[0]),
        "target_y": float(target[1]),
        "target_yaw": float(target[2]),
        "target_dx": float(target[0] - cxy[0]),
        "target_dy": float(target[1] - cxy[1]),
        "target_yaw_error": wrap_angle(float(target[2]) - trailer_yaw(model, data)),
        "desired_hitch_x": float(desired_hitch[0]),
        "desired_hitch_y": float(desired_hitch[1]),
        "desired_hitch_dx": float(desired_hitch[0] - hxy[0]),
        "desired_hitch_dy": float(desired_hitch[1] - hxy[1]),
        "trailer_tail_x": float(tail[0]),
        "trailer_tail_y": float(tail[1]),
        "tractor_nose_x": float(nose[0]),
        "tractor_nose_y": float(nose[1]),
        "reverse_required": bool(scenario.get("reverse_required", False)),
        "trailer_length": float(scenario.get("trailer_length", DEFAULT_TRAILER_LENGTH)),
        "max_drive_speed": float(scenario.get("max_drive_speed", DEFAULT_MAX_DRIVE_SPEED)),
        "max_yaw_rate": float(scenario.get("max_yaw_rate", DEFAULT_MAX_YAW_RATE)),
        "workspace": scenario.get("workspace", DEFAULT_WORKSPACE),
        "no_go": scenario.get("no_go", []),
    }


def clip_action(action: Any) -> np.ndarray:
    """Return a finite normalized two-element action clipped to [-1, 1]."""
    try:
        drive, steer = action
    except Exception as exc:  # noqa: BLE001
        raise ValueError("action must be a two-element sequence") from exc
    values = np.array([float(drive), float(steer)], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("action values must be finite")
    return np.clip(values, -1.0, 1.0)


def apply_disturbance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: dict[str, Any],
    time_sec: float,
) -> None:
    """Apply a pinned deterministic disturbance when requested."""
    disturbance = scenario.get("disturbance")
    if not disturbance:
        return
    dt = float(model.opt.timestep)
    disturbance_step = int(math.floor(float(disturbance.get("time", -1.0)) / dt + 0.5))
    current_step = int(math.floor(float(time_sec) / dt + 0.5))
    if current_step != disturbance_step:
        return
    idx = indices(model)
    hv = disturbance.get("hitch_velocity", [0.0, 0.0])
    data.qvel[idx["hitch_x_qvel"]] += float(hv[0])
    data.qvel[idx["hitch_y_qvel"]] += float(hv[1])
    data.qvel[idx["trailer_hinge_qvel"]] += float(disturbance.get("trailer_yaw_rate", 0.0))
    data.qpos[idx["hitch_x_qpos"]] += float(hv[0]) * dt
    data.qpos[idx["hitch_y_qpos"]] += float(hv[1]) * dt
    data.qpos[idx["trailer_hinge_qpos"]] = wrap_angle(
        data.qpos[idx["trailer_hinge_qpos"]] + float(disturbance.get("trailer_yaw_rate", 0.0)) * dt
    )


def kinematic_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: dict[str, Any],
    action: Any,
    time_sec: float,
    *,
    advance_time: bool = True,
) -> np.ndarray:
    """Advance the deterministic nonholonomic trailer state by one step."""
    clipped = clip_action(action)
    idx = indices(model)
    dt = float(model.opt.timestep)
    length = float(scenario.get("trailer_length", DEFAULT_TRAILER_LENGTH))
    speed_scale = float(scenario.get("speed_scale", 1.0))
    yaw_scale = float(scenario.get("yaw_scale", 1.0))
    max_drive = float(scenario.get("max_drive_speed", DEFAULT_MAX_DRIVE_SPEED)) * speed_scale
    max_yaw = float(scenario.get("max_yaw_rate", DEFAULT_MAX_YAW_RATE)) * yaw_scale
    drive = float(clipped[0]) * max_drive
    yaw_rate = float(clipped[1]) * max_yaw

    psi = tractor_yaw(model, data)
    phi = hitch_angle(model, data)
    x_dot = drive * math.cos(psi)
    y_dot = drive * math.sin(psi)
    theta_dot = -(drive / max(length, 1e-6)) * math.sin(phi)
    phi_dot = theta_dot - yaw_rate

    data.qpos[idx["hitch_x_qpos"]] += x_dot * dt
    data.qpos[idx["hitch_y_qpos"]] += y_dot * dt
    data.qpos[idx["tractor_yaw_qpos"]] = wrap_angle(data.qpos[idx["tractor_yaw_qpos"]] + yaw_rate * dt)
    data.qpos[idx["trailer_hinge_qpos"]] = wrap_angle(data.qpos[idx["trailer_hinge_qpos"]] + phi_dot * dt)
    data.qvel[idx["hitch_x_qvel"]] = x_dot
    data.qvel[idx["hitch_y_qvel"]] = y_dot
    data.qvel[idx["tractor_yaw_qvel"]] = yaw_rate
    data.qvel[idx["trailer_hinge_qvel"]] = phi_dot
    apply_disturbance(model, data, scenario, time_sec)
    if advance_time:
        data.time = float(time_sec) + dt
    mujoco.mj_forward(model, data)
    return clipped


def no_go_clearance(point: np.ndarray, no_go: list[dict[str, Any]], radius: float = SAFETY_RADIUS) -> float:
    if not no_go:
        return 1.0
    clearances: list[float] = []
    for item in no_go:
        if item.get("type") != "circle":
            continue
        center = np.array(item["center"], dtype=float)
        clearances.append(float(np.linalg.norm(point - center) - float(item["radius"]) - radius))
    return min(clearances) if clearances else 1.0


def workspace_margin(point: np.ndarray, workspace: dict[str, float] | None = None, radius: float = SAFETY_RADIUS) -> float:
    ws = workspace or DEFAULT_WORKSPACE
    return min(
        point[0] - float(ws["x_min"]) - radius,
        float(ws["x_max"]) - point[0] - radius,
        point[1] - float(ws["y_min"]) - radius,
        float(ws["y_max"]) - point[1] - radius,
    )


def scenario_observation_schema() -> dict[str, str]:
    """Return the public observation fields for documentation/tests."""
    return {
        "hitch_x/hitch_y": "tractor hitch position",
        "tractor_yaw": "tractor heading in radians",
        "trailer_x/trailer_y/trailer_yaw": "trailer center pose",
        "hitch_angle": "relative trailer-minus-tractor yaw",
        "target_x/target_y/target_yaw": "desired trailer center pose",
        "desired_hitch_x/desired_hitch_y": "hitch pose implied by target trailer pose",
        "reverse_required": "whether controlled backing is expected",
        "workspace": "workspace bounds",
        "no_go": "hidden/public circular no-go regions",
        "max_drive_speed/max_yaw_rate": "action scaling for normalized commands",
    }
