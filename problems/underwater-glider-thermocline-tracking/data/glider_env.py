from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np

GLIDER_LENGTH = 0.105
GLIDER_RADIUS = 0.030
DEFAULT_DT = 0.030
DEFAULT_WORKSPACE = {"x_min": -0.12, "x_max": 2.15, "z_min": 0.18, "z_max": 1.42}


def _xml_escape(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;")


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(value)))


def target_depth_at(scenario: dict[str, Any], x_pos: float) -> float:
    profile = scenario.get("thermocline", {})
    depth = float(profile.get("base", 0.78)) + float(profile.get("slope", 0.0)) * float(x_pos)
    for wave in profile.get("waves", []):
        depth += float(wave.get("amp", 0.0)) * math.sin(
            float(wave.get("freq", 1.0)) * float(x_pos) + float(wave.get("phase", 0.0))
        )
    bounds = scenario.get("workspace", DEFAULT_WORKSPACE)
    return _clamp(depth, float(bounds["z_min"]) + 0.08, float(bounds["z_max"]) - 0.08)


def target_slope_at(scenario: dict[str, Any], x_pos: float) -> float:
    profile = scenario.get("thermocline", {})
    slope = float(profile.get("slope", 0.0))
    for wave in profile.get("waves", []):
        amp = float(wave.get("amp", 0.0))
        freq = float(wave.get("freq", 1.0))
        phase = float(wave.get("phase", 0.0))
        slope += amp * freq * math.cos(freq * float(x_pos) + phase)
    return slope


def _sample_center(scenario: dict[str, Any], sample: dict[str, Any]) -> np.ndarray:
    x_pos = float(sample["x"])
    z_pos = float(sample.get("z", target_depth_at(scenario, x_pos)))
    return np.array([x_pos, z_pos], dtype=float)


def build_model(scenario: dict[str, Any]) -> mujoco.MjModel:
    """Build a compact MuJoCo model used as a deterministic visual state."""
    workspace = scenario.get("workspace", DEFAULT_WORKSPACE)
    x_mid = 0.5 * (float(workspace["x_min"]) + float(workspace["x_max"]))
    z_mid = 0.5 * (float(workspace["z_min"]) + float(workspace["z_max"]))
    sx = 0.5 * (float(workspace["x_max"]) - float(workspace["x_min"]))
    sz = 0.5 * (float(workspace["z_max"]) - float(workspace["z_min"]))

    geoms: list[str] = [
        f'<geom name="water_window" type="box" pos="{x_mid:.4f} {z_mid:.4f} -0.012" '
        f'size="{sx:.4f} {sz:.4f} 0.010" rgba="0.02 0.12 0.20 1" contype="0" conaffinity="0"/>',
    ]

    profile_points = []
    for idx in range(30):
        x_pos = float(workspace["x_min"]) + (float(workspace["x_max"]) - float(workspace["x_min"])) * idx / 29.0
        profile_points.append((x_pos, target_depth_at(scenario, x_pos)))
    for idx, (x_pos, z_pos) in enumerate(profile_points):
        geoms.append(
            f'<geom name="thermocline_{idx}" type="sphere" pos="{x_pos:.4f} {z_pos:.4f} 0.014" '
            'size="0.014" rgba="0.20 0.86 1.00 0.58" contype="0" conaffinity="0"/>'
        )

    for index, sample in enumerate(scenario.get("samples", [])):
        x_pos, z_pos = _sample_center(scenario, sample)
        radius_x = float(sample.get("radius_x", 0.060))
        radius_z = float(sample.get("radius_z", 0.070))
        geoms.append(
            f'<geom name="sample_{index}" type="ellipsoid" pos="{x_pos:.4f} {z_pos:.4f} 0.018" '
            f'size="{radius_x:.4f} {radius_z:.4f} 0.006" rgba="0.08 0.95 0.42 0.45" '
            'contype="0" conaffinity="0"/>'
        )

    for index, plume in enumerate(scenario.get("plumes", [])):
        cx, cz = plume["center"]
        radius = float(plume["radius"])
        geoms.append(
            f'<geom name="plume_{index}" type="cylinder" pos="{float(cx):.4f} {float(cz):.4f} 0.020" '
            f'size="{radius:.4f} 0.018" rgba="0.96 0.20 0.08 0.50" contype="0" conaffinity="0"/>'
        )

    xml = f"""
<mujoco model="{_xml_escape(str(scenario.get("id", "underwater_glider")))}">
  <compiler angle="radian"/>
  <option timestep="{float(scenario.get("dt", DEFAULT_DT)):.6f}" gravity="0 0 0" integrator="Euler"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>
  <worldbody>
    <light pos="0.8 -0.4 1.8" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <camera name="track" pos="1.0 0.8 2.3" xyaxes="1 0 0 0 1 0"/>
    {"".join(geoms)}
    <body name="glider" pos="0 0 0.055">
      <joint name="x" type="slide" axis="1 0 0" damping="0"/>
      <joint name="depth" type="slide" axis="0 1 0" damping="0"/>
      <joint name="pitch" type="hinge" axis="0 0 1" damping="0"/>
      <geom name="body" type="capsule" fromto="-0.052 0 0 0.052 0 0" size="{GLIDER_RADIUS:.4f}" rgba="1.00 0.76 0.18 1"/>
      <geom name="nose" type="sphere" pos="0.064 0 0" size="0.024" rgba="1.00 0.92 0.40 1"/>
      <geom name="fin_left" type="box" pos="-0.018 0.045 0" size="0.045 0.010 0.005" rgba="0.75 0.86 0.95 1"/>
      <geom name="fin_right" type="box" pos="-0.018 -0.045 0" size="0.045 0.010 0.005" rgba="0.75 0.86 0.95 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    return mujoco.MjModel.from_xml_string(xml)


def reset_data(model: mujoco.MjModel, scenario: dict[str, Any]) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    start = np.array(scenario["start"], dtype=float)
    data.qpos[:2] = start
    data.qpos[2] = float(scenario.get("initial_pitch", 0.0))
    data.qvel[:2] = np.array(scenario.get("initial_velocity", [0.0, 0.0]), dtype=float)
    data.qvel[2] = 0.0
    data.time = 0.0
    mujoco.mj_forward(model, data)
    return data


def current_at(scenario: dict[str, Any], point: np.ndarray, time_sec: float) -> np.ndarray:
    x_pos, z_pos = float(point[0]), float(point[1])
    current = np.array(scenario.get("base_current", [0.0, 0.0]), dtype=float)
    shear = scenario.get("shear", {})
    if shear:
        amp = float(shear.get("amplitude", 0.0))
        freq = float(shear.get("frequency", 1.0))
        phase = float(shear.get("phase", 0.0))
        current += np.array(
            [
                amp * math.sin(freq * z_pos + phase + 0.22 * time_sec),
                0.42 * amp * math.sin(0.75 * freq * x_pos - phase + 0.18 * time_sec),
            ],
            dtype=float,
        )
    for eddy in scenario.get("eddies", []):
        cx, cz = eddy["center"]
        dx = x_pos - float(cx)
        dz = z_pos - float(cz)
        r2 = dx * dx + dz * dz + 0.030
        strength = float(eddy.get("strength", 0.0))
        current += strength * np.array([-dz, dx], dtype=float) / r2
    max_current = float(scenario.get("max_current", 0.26))
    norm = float(np.linalg.norm(current))
    if norm > max_current:
        current *= max_current / norm
    return current


def clip_action(action: Any) -> np.ndarray:
    try:
        pitch_cmd, buoyancy_cmd = action
    except Exception as exc:  # noqa: BLE001
        raise ValueError("action must be [pitch_command, buoyancy_command]") from exc
    values = np.array([float(pitch_cmd), float(buoyancy_cmd)], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("action values must be finite")
    return np.clip(values, -1.0, 1.0)


def kinematic_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: dict[str, Any],
    action: Any,
    time_sec: float,
    *,
    advance_time: bool = True,
) -> np.ndarray:
    action_vec = clip_action(action)
    dt = float(model.opt.timestep)
    point = np.array(data.qpos[:2], dtype=float)
    velocity = np.array(data.qvel[:2], dtype=float)
    pitch = float(data.qpos[2])
    pitch_rate = float(data.qvel[2])

    max_pitch = float(scenario.get("max_pitch", 0.70))
    max_pitch_rate = float(scenario.get("max_pitch_rate", 1.65))
    pitch_rate += (max_pitch_rate * action_vec[0] - 2.0 * pitch_rate) * dt
    pitch = _clamp(pitch + pitch_rate * dt, -max_pitch, max_pitch)
    if abs(pitch) >= max_pitch - 1e-9 and pitch * pitch_rate > 0:
        pitch_rate = 0.0

    base_speed = float(scenario.get("base_speed", 0.205))
    speed_response = float(scenario.get("speed_response", 1.75))
    target_vx = base_speed * (0.78 + 0.22 * math.cos(pitch))
    target_vz = float(scenario.get("glide_gain", 0.44)) * math.sin(pitch)
    target_vz += float(scenario.get("buoyancy_gain", 0.38)) * action_vec[1]
    desired = np.array([target_vx, target_vz], dtype=float)
    velocity += speed_response * (desired - velocity) * dt

    max_speed = float(scenario.get("max_speed", 0.42))
    norm = float(np.linalg.norm(velocity))
    if norm > max_speed:
        velocity *= max_speed / norm
    current = current_at(scenario, point, time_sec)
    point = point + (velocity + current) * dt

    data.qpos[0] = point[0]
    data.qpos[1] = point[1]
    data.qpos[2] = pitch
    data.qvel[0] = velocity[0]
    data.qvel[1] = velocity[1]
    data.qvel[2] = pitch_rate
    if advance_time:
        data.time = time_sec + dt
    mujoco.mj_forward(model, data)
    return action_vec


def sample_reached(point: np.ndarray, scenario: dict[str, Any], sample: dict[str, Any]) -> bool:
    center = _sample_center(scenario, sample)
    rx = float(sample.get("radius_x", 0.060))
    rz = float(sample.get("radius_z", 0.070))
    value = ((float(point[0]) - center[0]) / rx) ** 2 + ((float(point[1]) - center[1]) / rz) ** 2
    return value <= 1.0


def workspace_margin(point: np.ndarray, workspace: dict[str, float] | None) -> float:
    ws = workspace or DEFAULT_WORKSPACE
    return min(
        float(point[0]) - float(ws.get("x_min", DEFAULT_WORKSPACE["x_min"])),
        float(ws.get("x_max", DEFAULT_WORKSPACE["x_max"])) - float(point[0]),
        float(point[1]) - float(ws.get("z_min", DEFAULT_WORKSPACE["z_min"])),
        float(ws.get("z_max", DEFAULT_WORKSPACE["z_max"])) - float(point[1]),
    ) - GLIDER_RADIUS


def plume_clearance(point: np.ndarray, scenario: dict[str, Any]) -> float:
    best = 10.0
    for plume in scenario.get("plumes", []):
        center = np.array(plume["center"], dtype=float)
        clearance = float(np.linalg.norm(point - center)) - float(plume["radius"]) - GLIDER_RADIUS
        best = min(best, clearance)
    return best


def observation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scenario: dict[str, Any],
    time_sec: float,
    sample_index: int,
    sample_hold_progress: float = 0.0,
) -> dict[str, Any]:
    point = np.array(data.qpos[:2], dtype=float)
    velocity = np.array(data.qvel[:2], dtype=float)
    samples = scenario.get("samples", [])
    if sample_index < len(samples):
        active = samples[sample_index]
        goal = _sample_center(scenario, active)
        goal_kind = "sample"
        sample_radius_x = float(active.get("radius_x", 0.060))
        sample_radius_z = float(active.get("radius_z", 0.070))
    else:
        goal = np.array(scenario["finish"], dtype=float)
        goal_kind = "finish"
        sample_radius_x = 0.0
        sample_radius_z = 0.0
    current = current_at(scenario, point, time_sec)
    target_depth = target_depth_at(scenario, float(point[0]))
    target_slope = target_slope_at(scenario, float(point[0]))
    return {
        "time": float(time_sec),
        "dt": float(model.opt.timestep),
        "duration": float(scenario.get("duration", 10.0)),
        "x": float(point[0]),
        "z": float(point[1]),
        "vx": float(velocity[0]),
        "vz": float(velocity[1]),
        "pitch": float(data.qpos[2]),
        "pitch_rate": float(data.qvel[2]),
        "current_x": float(current[0]),
        "current_z": float(current[1]),
        "target_depth": float(target_depth),
        "target_slope": float(target_slope),
        "goal_kind": goal_kind,
        "goal_x": float(goal[0]),
        "goal_z": float(goal[1]),
        "sample_index": int(sample_index),
        "num_samples": len(samples),
        "sample_radius_x": sample_radius_x,
        "sample_radius_z": sample_radius_z,
        "sample_hold_progress": float(sample_hold_progress),
        "sample_hold_time": float(scenario.get("sample_hold_time", 0.18)),
        "finish_x": float(scenario["finish"][0]),
        "finish_z": float(scenario["finish"][1]),
        "glider_radius": GLIDER_RADIUS,
        "workspace": scenario.get("workspace", DEFAULT_WORKSPACE),
        "plumes": scenario.get("plumes", []),
        "max_pitch": float(scenario.get("max_pitch", 0.70)),
        "max_speed": float(scenario.get("max_speed", 0.42)),
    }
