"""Deterministic hidden-scenario scorer for planar snake gate navigation."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from grading import PolicyWorker, PolicyWorkerError

DATA_DIRS = [
    Path("/data"),
    Path(__file__).resolve().parents[1] / "data",
]
for data_dir in DATA_DIRS:
    if data_dir.exists() and str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))

from snake_env import (  # noqa: E402
    ACTION_SIZE,
    LINK_RADIUS,
    NUM_JOINTS,
    active_gate,
    apply_action,
    apply_disturbance,
    body_points,
    build_model,
    gate_local_error,
    gate_passed,
    head_xy,
    head_yaw,
    indices,
    no_go_clearance,
    observation,
    reset_data,
    tail_xy,
    workspace_margin,
    wrap_angle,
)

ACCEPTANCE_CUTOFF = 0.40
ORACLE_RAW_HEADLINE = 0.6599347862038533

CRITERION_DESCRIPTIONS = {
    "policy_present": "Submitted /tmp/output/policy.py imports and exposes act(obs), get_action(obs), or Policy.act(obs).",
    "gate_progress": "Sequential hidden gate completion by the snake head; full credit requires passing every gate in order.",
    "gate_accuracy": "Mean closest approach to each hidden gate center and opening; full credit near 0.08 m, zero by 0.48 m.",
    "final_target": "Final-window head distance to the last target; full credit at 0.11 m, zero by 0.72 m.",
    "heading_control": "Final-window heading alignment to the active gate direction; full credit below 0.18 rad, zero by 1.25 rad.",
    "body_clearance": "Whole-body workspace and no-go-circle clearance across all link samples; workspace credit ramps from -0.25 m to 0.04 m and no-go credit from -0.16 m to 0.03 m.",
    "body_coherence": "Finite articulated motion with max joint angle below 1.22 rad, joint speed below 9.0 rad/s, and head speed below 1.65 m/s.",
    "gait_activity": "Non-degenerate articulated gait: joint RMS should stay in a moderate 0.05-0.92 rad band with adjacent-link phase variation.",
    "smoothness": "Mean action magnitude and action-change penalty normalized by action dimension; full credit near 0.32 mean action and 0.10 mean delta.",
    "scenario_completion": "Aggregate per-scenario guard: minimum of progress, accuracy, final target, clearance, coherence, and gait activity.",
    "worst_case": "Worst hidden-scenario completion score, rewarding policies that handle every gate family.",
}

SCENARIO_WEIGHTS = {
    "gate_progress": 0.22,
    "gate_accuracy": 0.11,
    "final_target": 0.13,
    "heading_control": 0.06,
    "body_clearance": 0.15,
    "body_coherence": 0.06,
    "gait_activity": 0.15,
    "smoothness": 0.04,
    "scenario_completion": 0.08,
}
AVERAGE_SCENARIO_WEIGHT = 0.60
WORST_CASE_WEIGHT = 0.40


def _clamp01(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _progress_lower(value: float, floor: float, perfect: float) -> float:
    if floor <= perfect:
        return 0.0
    return _clamp01((floor - float(value)) / (floor - perfect))


def _progress_upper(value: float, floor: float, perfect: float) -> float:
    if perfect <= floor:
        return 0.0
    return _clamp01((float(value) - floor) / (perfect - floor))


def _band_score(value: float, low_floor: float, low_good: float, high_good: float, high_floor: float) -> float:
    return min(_progress_upper(value, low_floor, low_good), _progress_lower(value, high_floor, high_good))


def _calibrate_headline(raw_score: float) -> float:
    raw_score = _clamp01(raw_score)
    if raw_score <= ACCEPTANCE_CUTOFF:
        return raw_score
    if ORACLE_RAW_HEADLINE <= ACCEPTANCE_CUTOFF:
        return raw_score
    return _clamp01(
        ACCEPTANCE_CUTOFF
        + (raw_score - ACCEPTANCE_CUTOFF) * ((1.0 - ACCEPTANCE_CUTOFF) / (ORACLE_RAW_HEADLINE - ACCEPTANCE_CUTOFF))
    )


def _rubric_rows(subscores: dict[str, float], weights: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, score in subscores.items():
        description = CRITERION_DESCRIPTIONS.get(key, key)
        rows.append(
            {
                "name": key,
                "label": key,
                "criterion": key,
                "id": key,
                "criterion_id": key,
                "description": description,
                "score": float(score),
                "max_score": 1.0,
                "weight": float(weights.get(key, 0.0)),
                "reasoning": "",
                "grading_criteria": description,
            }
        )
    return rows


def _failed_scenario(scenario: dict[str, Any], error: str) -> dict[str, Any]:
    result = {
        "id": scenario.get("id", "unknown"),
        "family": scenario.get("family", "unknown"),
        "score": 0.0,
        "error": error,
        "finite": 0.0,
        "gate_count": len(scenario.get("gates", [])),
        "passed_gates": 0,
        "final_distance": 999.0,
        "tail_distance": 999.0,
        "final_heading_error": math.pi,
        "min_workspace_margin": -1.0,
        "min_no_go_clearance": -1.0,
        "joint_rms": 0.0,
        "adjacent_phase_mean": 0.0,
        "max_abs_joint": 0.0,
        "max_joint_speed": 0.0,
        "max_head_speed": 0.0,
        "mean_action": 0.0,
        "mean_delta_action": 0.0,
    }
    for key in SCENARIO_WEIGHTS:
        result[key] = 0.0
    return result


class _PolicyCaller:
    """Call submitted policies through the narrow PolicyWorker JSON API."""

    def __init__(self, worker: PolicyWorker) -> None:
        self.worker = worker
        self.method: str | None = None

    def __call__(self, obs: dict[str, Any]) -> Any:
        if self.method is not None:
            return self.worker.call(self.method, obs)
        try:
            result = self.worker.call("act", obs)
        except PolicyWorkerError as exc:
            message = str(exc)
            missing_act = "has no attribute 'act'" in message or 'has no attribute \"act\"' in message
            if not missing_act:
                raise
        else:
            self.method = "act"
            return result
        result = self.worker.call("get_action", obs)
        self.method = "get_action"
        return result


def _scenario_score(policy: _PolicyCaller, scenario: dict[str, Any]) -> dict[str, Any]:
    model = build_model(scenario)
    data = reset_data(model, scenario)
    idx = indices(model)
    duration = float(scenario.get("duration", 8.0))
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    gates = list(scenario.get("gates", []))
    gate_index = 0
    final_target_xy = np.array(scenario.get("target", gates[-1]["center"] if gates else [0.0, 0.0]), dtype=float)
    workspace = scenario.get("workspace")
    no_go = list(scenario.get("no_go", []))

    gate_min_dist = [10.0 for _ in gates]
    gate_min_lateral = [10.0 for _ in gates]
    heading_errors: list[float] = []
    final_distances: list[float] = []
    final_heading_errors: list[float] = []
    actions: list[np.ndarray] = []
    joint_angles: list[np.ndarray] = []
    joint_velocities: list[np.ndarray] = []
    head_speeds: list[float] = []
    min_workspace_margin = 10.0
    min_no_go = 10.0
    finite = True
    error: str | None = None

    for step in range(steps):
        time_sec = step * dt
        hxy = head_xy(model, data, idx)
        for gate_id, gate in enumerate(gates):
            _longitudinal, lateral, distance = gate_local_error(hxy, gate)
            gate_min_dist[gate_id] = min(gate_min_dist[gate_id], distance)
            gate_min_lateral[gate_id] = min(gate_min_lateral[gate_id], abs(lateral))

        while gate_index < len(gates) and gate_passed(hxy, gates[gate_index]):
            gate_index += 1

        obs = observation(model, data, scenario, time_sec, gate_index, idx)
        try:
            action = apply_action(model, data, policy(obs), scenario)
        except Exception as exc:  # noqa: BLE001
            finite = False
            error = f"policy_error: {exc}"
            break
        actions.append(action)
        apply_disturbance(model, data, scenario, time_sec)
        mujoco.mj_step(model, data)

        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            finite = False
            error = "non-finite MuJoCo state"
            break

        points = body_points(model, data, idx)
        for point in points:
            min_workspace_margin = min(min_workspace_margin, workspace_margin(point, workspace, LINK_RADIUS))
            min_no_go = min(min_no_go, no_go_clearance(point, no_go, LINK_RADIUS))

        joint_angles.append(np.asarray(data.qpos[idx["joint_qpos"]], dtype=float).copy())
        joint_velocities.append(np.asarray(data.qvel[idx["joint_qvel"]], dtype=float).copy())
        speed = float(np.linalg.norm([data.qvel[0], data.qvel[1]]))
        head_speeds.append(speed)

        active = active_gate(scenario, gate_index)
        target_yaw = float(active.get("yaw", 0.0))
        heading_error = abs(wrap_angle(target_yaw - head_yaw(model, data, idx)))
        heading_errors.append(heading_error)

        if step >= steps - max(1, int(0.9 / dt)):
            final_distances.append(float(np.linalg.norm(head_xy(model, data, idx) - final_target_xy)))
            final_heading_errors.append(heading_error)

    if not actions:
        return _failed_scenario(scenario, error or "no rollout samples")
    if not finite:
        return _failed_scenario(scenario, error or "invalid rollout")

    while gate_index < len(gates) and gate_passed(head_xy(model, data, idx), gates[gate_index]):
        gate_index += 1

    if gates:
        gate_progress = gate_index / len(gates)
        gate_distance_score = float(np.mean([_progress_lower(value, floor=0.48, perfect=0.08) for value in gate_min_dist]))
        gate_lateral_score = float(
            np.mean(
                [
                    _progress_lower(
                        gate_min_lateral[i],
                        floor=0.5 * float(gates[i].get("width", 0.28)) + 0.18,
                        perfect=0.5 * float(gates[i].get("width", 0.28)) * 0.45,
                    )
                    for i in range(len(gates))
                ]
            )
        )
    else:
        gate_progress = 1.0
        gate_distance_score = 1.0
        gate_lateral_score = 1.0

    final_distance = float(np.mean(final_distances or [np.linalg.norm(head_xy(model, data, idx) - final_target_xy)]))
    final_heading = float(np.mean(final_heading_errors or heading_errors or [0.0]))
    joint_array = np.asarray(joint_angles, dtype=float) if joint_angles else np.zeros((1, NUM_JOINTS))
    joint_vel_array = np.asarray(joint_velocities, dtype=float) if joint_velocities else np.zeros((1, NUM_JOINTS))
    action_array = np.asarray(actions, dtype=float)

    joint_rms = float(np.sqrt(np.mean(np.square(joint_array))))
    adjacent_phase = float(np.mean(np.abs(np.diff(joint_array, axis=1)))) if joint_array.shape[1] > 1 else 0.0
    max_abs_joint = float(np.max(np.abs(joint_array)))
    max_joint_speed = float(np.max(np.abs(joint_vel_array)))
    max_head_speed = float(max(head_speeds or [0.0]))
    mean_action = float(np.mean(np.linalg.norm(action_array, axis=1))) / math.sqrt(ACTION_SIZE)
    mean_du = (
        float(np.mean(np.linalg.norm(np.diff(action_array, axis=0), axis=1))) / math.sqrt(ACTION_SIZE)
        if len(actions) > 1
        else 0.0
    )

    tail_distance = float(np.linalg.norm(tail_xy(model, data, idx) - final_target_xy))
    gate_accuracy = 0.65 * gate_distance_score + 0.35 * gate_lateral_score
    final_target_score = _progress_lower(final_distance, floor=0.72, perfect=0.11)
    heading_control = _progress_lower(final_heading, floor=1.25, perfect=0.18)
    workspace_score = _progress_upper(min_workspace_margin, floor=-0.25, perfect=0.04)
    no_go_score = _progress_upper(min_no_go, floor=-0.16, perfect=0.03)
    body_clearance = min(workspace_score, no_go_score)
    body_coherence = min(
        1.0 if finite else 0.0,
        _progress_lower(max_abs_joint, floor=1.22, perfect=0.92),
        _progress_lower(max_joint_speed, floor=9.0, perfect=4.4),
        _progress_lower(max_head_speed, floor=1.65, perfect=0.85),
    )
    gait_activity = 0.72 * _band_score(joint_rms, low_floor=0.05, low_good=0.22, high_good=0.58, high_floor=0.92) + 0.28 * _band_score(
        adjacent_phase,
        low_floor=0.02,
        low_good=0.15,
        high_good=0.62,
        high_floor=1.05,
    )
    smoothness = 0.55 * _progress_lower(mean_action, floor=0.95, perfect=0.32) + 0.45 * _progress_lower(
        mean_du, floor=0.72, perfect=0.10
    )
    scenario_completion = min(
        gate_progress,
        gate_accuracy,
        final_target_score,
        body_clearance,
        body_coherence,
        gait_activity,
    )

    scenario_subscores = {
        "gate_progress": _clamp01(gate_progress),
        "gate_accuracy": _clamp01(gate_accuracy),
        "final_target": _clamp01(final_target_score),
        "heading_control": _clamp01(heading_control),
        "body_clearance": _clamp01(body_clearance),
        "body_coherence": _clamp01(body_coherence),
        "gait_activity": _clamp01(gait_activity),
        "smoothness": _clamp01(smoothness),
        "scenario_completion": _clamp01(scenario_completion),
    }
    score = sum(SCENARIO_WEIGHTS[key] * scenario_subscores[key] for key in SCENARIO_WEIGHTS)
    return {
        "id": scenario.get("id", "unknown"),
        "family": scenario.get("family", "unknown"),
        "score": _clamp01(score),
        "finite": 1.0 if finite else 0.0,
        **scenario_subscores,
        "gate_count": len(gates),
        "passed_gates": gate_index,
        "final_distance": final_distance,
        "tail_distance": tail_distance,
        "final_heading_error": final_heading,
        "min_workspace_margin": min_workspace_margin,
        "min_no_go_clearance": min_no_go,
        "joint_rms": joint_rms,
        "adjacent_phase_mean": adjacent_phase,
        "max_abs_joint": max_abs_joint,
        "max_joint_speed": max_joint_speed,
        "max_head_speed": max_head_speed,
        "mean_action": mean_action,
        "mean_delta_action": mean_du,
        "error": error,
    }


def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | None,
    private: Path,
) -> dict[str, Any]:
    """Score a submitted snake controller against hidden deterministic gate layouts."""

    _ = trajectory
    policy_path = workspace / "policy.py"
    if not policy_path.exists():
        return {
            "score": 0.0,
            "subscores": {"policy_present": 0.0},
            "weights": {"policy_present": 1.0},
            "metadata": {"error": "missing /tmp/output/policy.py"},
        }

    try:
        scenarios = json.loads((private / "hidden_scenarios.json").read_text())
        scenario_results: list[dict[str, Any]] = []
        worker_cwd = next((path for path in DATA_DIRS if path.exists()), workspace)
        for scenario in scenarios:
            with PolicyWorker(policy_path, timeout_s=0.35, cwd=worker_cwd) as worker:
                scenario_results.append(_scenario_score(_PolicyCaller(worker), scenario))
    except Exception as exc:  # noqa: BLE001
        return {
            "score": 0.0,
            "subscores": {"policy_present": 1.0, "rollout_valid": 0.0},
            "weights": {"policy_present": 0.0, "rollout_valid": 1.0},
            "metadata": {"error": str(exc)},
        }

    scores = np.array([result["score"] for result in scenario_results], dtype=float)
    avg_score = float(np.mean(scores)) if len(scores) else 0.0
    worst_score = float(np.min(scores)) if len(scores) else 0.0
    worst_completion = (
        float(np.min([result["scenario_completion"] for result in scenario_results]))
        if scenario_results
        else 0.0
    )
    raw_headline = _clamp01(AVERAGE_SCENARIO_WEIGHT * avg_score + WORST_CASE_WEIGHT * worst_completion)
    headline = _calibrate_headline(raw_headline)

    subscore_keys = list(SCENARIO_WEIGHTS)
    subscores = {
        key: float(np.mean([result[key] for result in scenario_results]))
        for key in subscore_keys
    }
    subscores["policy_present"] = 1.0
    subscores["worst_case"] = worst_completion
    weights = {
        "policy_present": 0.0,
        **{key: AVERAGE_SCENARIO_WEIGHT * weight for key, weight in SCENARIO_WEIGHTS.items()},
        "worst_case": WORST_CASE_WEIGHT,
    }
    rubric_rows = _rubric_rows(subscores, weights)
    return {
        "score": headline,
        "subscores": subscores,
        "weights": weights,
        "structured_subscores": rubric_rows,
        "metadata": {
            "num_scenarios": len(scenario_results),
            "raw_headline_score": raw_headline,
            "headline_score": headline,
            "reported_final_score": headline,
            "oracle_reference_raw_headline": ORACLE_RAW_HEADLINE,
            "calibration_note": "Scores at or below the acceptance cutoff are unchanged; oracle-level raw scores are normalized to 1.0.",
            "acceptance_cutoff_unchanged_below": ACCEPTANCE_CUTOFF,
            "avg_scenario_score": avg_score,
            "worst_scenario_score": worst_score,
            "worst_completion_score": worst_completion,
            "scenario_details_redacted": True,
            "rubric_breakdown": rubric_rows,
            "diagnostics": {
                "finite_mean": float(np.mean([result["finite"] for result in scenario_results])) if scenario_results else 0.0,
                "passed_gates_mean": float(np.mean([result["passed_gates"] for result in scenario_results])) if scenario_results else 0.0,
                "joint_rms_mean": float(np.mean([result["joint_rms"] for result in scenario_results])) if scenario_results else 0.0,
                "min_no_go_clearance_min": float(np.min([result["min_no_go_clearance"] for result in scenario_results])) if scenario_results else 0.0,
            },
        },
    }
