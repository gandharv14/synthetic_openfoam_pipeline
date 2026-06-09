"""Deterministic rollout scorer for nonholonomic trailer docking."""

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
POLICY_CWD = next((data_dir for data_dir in DATA_DIRS if data_dir.exists()), None)

from trailer_env import (  # noqa: E402
    SAFETY_RADIUS,
    clip_action,
    hitch_angle,
    hitch_xy,
    kinematic_step,
    no_go_clearance,
    observation,
    reset_data,
    trailer_center_xy,
    trailer_tail_xy,
    trailer_yaw,
    tractor_nose_xy,
    workspace_margin,
    wrap_angle,
    build_model,
)

CRITERION_DESCRIPTIONS = {
    "policy_present": "Submitted /tmp/output/policy.py imports and exposes act(obs), get_action(obs), or Policy.act(obs).",
    "position": "Safety-gated final-window trailer-center position error; full credit at 0.07 m, zero at 0.75 m.",
    "orientation": "Safety/progress-gated final-window trailer yaw error; full credit at 0.10 rad, zero at 1.10 rad.",
    "progress": "Safety-gated fraction of initial trailer-target distance closed; full credit at 78%, zero below 5%.",
    "hold": "Safety/progress-gated final hold over the last 0.80 s; full credit requires trailer speed below 0.04 m/s.",
    "hitch_safety": "Low-articulation jackknife avoidance: final hitch full at 0.10 rad/zero at 0.55 rad, post-recovery max full at 0.25 rad/zero at 0.58 rad, and unsafe-hitch fraction zero by 10%.",
    "workspace": "Minimum clearance of hitch, tractor nose, trailer center, and trailer tail from workspace bounds.",
    "no_go": "Minimum clearance of hitch, tractor nose, trailer center, and trailer tail from hidden no-go circles.",
    "reverse_control": "Safety-gated useful sustained reverse behavior: reverse fraction full above 82%/zero below 52%, reverse target-distance reduction full above 50%/zero below 16%, and safe reverse samples.",
    "smoothness": "Low action magnitude and low action-to-action changes across the rollout, with tight slew-rate credit for stable trailer backing.",
    "worst_case": "Worst hidden scenario rollout score, used as a moderate robustness check against solving only easy layouts.",
}

ACCEPTANCE_CUTOFF = 0.40
ORACLE_RAW_HEADLINE = 0.8561591890248603


def _clamp01(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def _progress_lower(value: float, floor: float, perfect: float) -> float:
    if floor <= perfect:
        return 0.0
    return _clamp01((floor - value) / (floor - perfect))


def _progress_upper(value: float, floor: float, perfect: float) -> float:
    if perfect <= floor:
        return 0.0
    return _clamp01((value - floor) / (perfect - floor))


def _calibrate_headline(raw_score: float) -> float:
    """Keep scores below the acceptance cutoff unchanged and normalize the oracle to 1.0."""
    raw = _clamp01(raw_score)
    if raw <= ACCEPTANCE_CUTOFF:
        return raw
    if raw >= ORACLE_RAW_HEADLINE - 1e-12:
        return 1.0
    return _clamp01(
        ACCEPTANCE_CUTOFF
        + (1.0 - ACCEPTANCE_CUTOFF)
        * (raw - ACCEPTANCE_CUTOFF)
        / (ORACLE_RAW_HEADLINE - ACCEPTANCE_CUTOFF)
    )


def _rubric_rows(subscores: dict[str, float], weights: dict[str, float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, score in subscores.items():
        description = CRITERION_DESCRIPTIONS.get(key, key)
        rows.append(
            {
                "name": key,
                "label": key,
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


class _PolicyCaller:
    """Invoke submitted policies through PolicyWorker without exposing grader internals."""

    METHODS = ("act", "get_action")

    def __init__(self, worker: PolicyWorker) -> None:
        self.worker = worker
        self.method: str | None = None

    @staticmethod
    def _is_missing_method(exc: PolicyWorkerError, method: str) -> bool:
        message = str(exc)
        return f"has no attribute '{method}'" in message or f'has no attribute "{method}"' in message

    def __call__(self, obs: dict[str, Any]) -> Any:
        if self.method is not None:
            return self.worker.call(self.method, obs)

        # PolicyWorker normalizes module-level act(obs) and class Policy.act(obs)
        # to the same worker.call("act", obs) API. Probe all documented public
        # interfaces once, then cache the working method for the rollout.
        last_missing: PolicyWorkerError | None = None
        for method in self.METHODS:
            try:
                result = self.worker.call(method, obs)
            except PolicyWorkerError as exc:
                if not self._is_missing_method(exc, method):
                    raise
                last_missing = exc
                continue
            self.method = method
            return result

        if last_missing is not None:
            raise last_missing
        raise PolicyWorkerError("policy exposes no supported action method")


def _all_safety_points(model: mujoco.MjModel, data: mujoco.MjData, scenario: dict[str, Any]) -> list[np.ndarray]:
    return [
        hitch_xy(model, data),
        tractor_nose_xy(model, data),
        trailer_center_xy(model, data, scenario),
        trailer_tail_xy(model, data, scenario),
    ]


def _scenario_score(policy: _PolicyCaller, scenario: dict[str, Any]) -> dict[str, Any]:
    model = build_model(scenario)
    data = reset_data(model, scenario)
    target = np.array(scenario["target_pose"], dtype=float)
    initial_error = float(np.linalg.norm(trailer_center_xy(model, data, scenario) - target[:2]))
    duration = float(scenario.get("duration", 8.0))
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    final_window = max(1, int(0.80 / dt))
    recovery_window = int(0.80 / dt)

    actions: list[np.ndarray] = []
    final_pos_errors: list[float] = []
    final_yaw_errors: list[float] = []
    final_speeds: list[float] = []
    hitch_abs_after_recovery: list[float] = []
    reverse_steps = 0
    reverse_safe_steps = 0
    reverse_progress_m = 0.0
    min_workspace_margin = 10.0
    min_no_go_clearance = 10.0
    finite = True
    error: str | None = None

    prev_center = trailer_center_xy(model, data, scenario)
    max_trailer_speed = 0.0

    for step in range(steps):
        time_sec = step * dt
        obs = observation(model, data, scenario, time_sec)
        center_before = trailer_center_xy(model, data, scenario)
        error_before = float(np.linalg.norm(center_before - target[:2]))

        try:
            action = clip_action(policy(obs))
        except Exception as exc:  # noqa: BLE001
            finite = False
            error = f"policy_error: {exc}"
            break

        try:
            action = kinematic_step(model, data, scenario, action, time_sec)
        except Exception as exc:  # noqa: BLE001
            finite = False
            error = f"rollout_error: {exc}"
            break

        actions.append(action)
        reversing = bool(action[0] < -0.08)
        if reversing:
            reverse_steps += 1

        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            finite = False
            error = "non-finite MuJoCo state"
            break

        center = trailer_center_xy(model, data, scenario)
        error_after = float(np.linalg.norm(center - target[:2]))
        if reversing:
            reverse_progress_m += max(0.0, error_before - error_after)
        trailer_speed = float(np.linalg.norm(center - prev_center) / max(dt, 1e-9))
        prev_center = center.copy()
        max_trailer_speed = max(max_trailer_speed, trailer_speed)

        step_workspace_margin = 10.0
        step_no_go_clearance = 10.0
        for point in _all_safety_points(model, data, scenario):
            point_workspace_margin = workspace_margin(point, scenario.get("workspace"), SAFETY_RADIUS)
            point_no_go_clearance = no_go_clearance(point, scenario.get("no_go", []), SAFETY_RADIUS)
            step_workspace_margin = min(step_workspace_margin, point_workspace_margin)
            step_no_go_clearance = min(step_no_go_clearance, point_no_go_clearance)
            min_workspace_margin = min(min_workspace_margin, point_workspace_margin)
            min_no_go_clearance = min(min_no_go_clearance, point_no_go_clearance)

        if (
            reversing
            and abs(hitch_angle(model, data)) <= 1.05
            and step_workspace_margin >= -0.02
            and step_no_go_clearance >= -0.02
        ):
            reverse_safe_steps += 1

        if step >= recovery_window:
            hitch_abs_after_recovery.append(abs(hitch_angle(model, data)))

        if step >= steps - final_window:
            final_pos_errors.append(float(np.linalg.norm(center - target[:2])))
            final_yaw_errors.append(abs(wrap_angle(float(target[2]) - trailer_yaw(model, data))))
            final_speeds.append(trailer_speed)

    if not actions:
        return {
            "id": scenario.get("id", "unknown"),
            "family": scenario.get("family", "unknown"),
            "score": 0.0,
            "position": 0.0,
            "orientation": 0.0,
            "progress": 0.0,
            "hold": 0.0,
            "hitch_safety": 0.0,
            "workspace": 0.0,
            "no_go": 0.0,
            "reverse_control": 0.0,
            "smoothness": 0.0,
            "simultaneous_success": 0.0,
            "finite": 0.0,
            "achievement_gate": 0.0,
            "reverse_frac": 0.0,
            "reverse_progress_frac": 0.0,
            "reverse_safe_frac": 0.0,
            "error": error or "no rollout samples",
        }

    final_error = float(np.mean(final_pos_errors or [np.linalg.norm(trailer_center_xy(model, data, scenario) - target[:2])]))
    yaw_error = float(np.mean(final_yaw_errors or [abs(wrap_angle(float(target[2]) - trailer_yaw(model, data)))]))
    final_speed = float(np.mean(final_speeds or [0.0]))
    progress_m = max(0.0, initial_error - final_error)
    progress_frac = progress_m / max(initial_error, 1e-6)
    final_hitch_abs = abs(hitch_angle(model, data))
    max_hitch_abs = max(hitch_abs_after_recovery or [final_hitch_abs])
    unsafe_hitch_frac = float(np.mean([value > 1.05 for value in hitch_abs_after_recovery])) if hitch_abs_after_recovery else 0.0
    action_array = np.array(actions, dtype=float)
    mean_action = float(np.mean(np.linalg.norm(action_array, axis=1))) if len(action_array) else 0.0
    mean_du = (
        float(np.mean(np.linalg.norm(np.diff(action_array, axis=0), axis=1)))
        if len(action_array) > 1
        else 0.0
    )
    finite_score = 1.0 if finite else 0.0

    position_score = _progress_lower(final_error, floor=0.75, perfect=0.07)
    orientation_score = _progress_lower(yaw_error, floor=1.10, perfect=0.10)
    progress_score = _progress_upper(progress_frac, floor=0.05, perfect=0.78)
    speed_score = _progress_lower(final_speed, floor=0.42, perfect=0.04)
    hold_score = speed_score
    hitch_safety_score = min(
        _progress_lower(final_hitch_abs, floor=0.55, perfect=0.10),
        _progress_lower(max_hitch_abs, floor=0.58, perfect=0.25),
        _progress_lower(unsafe_hitch_frac, floor=0.10, perfect=0.0),
    )
    workspace_score = _progress_upper(min_workspace_margin, floor=-0.12, perfect=0.04)
    no_go_score = _progress_upper(min_no_go_clearance, floor=-0.10, perfect=0.05)
    reverse_frac = reverse_steps / max(1, len(actions))
    if scenario.get("reverse_required", False):
        reverse_fraction_score = _progress_upper(reverse_frac, floor=0.52, perfect=0.82)
        reverse_progress_frac = reverse_progress_m / max(initial_error, 1e-6)
        reverse_progress_score = _progress_upper(reverse_progress_frac, floor=0.16, perfect=0.50)
        reverse_safety_frac = reverse_safe_steps / max(1, reverse_steps)
        reverse_safety_score = _progress_upper(reverse_safety_frac, floor=0.82, perfect=0.98)
        reverse_score = (
            0.55 * reverse_fraction_score
            + 0.30 * reverse_progress_score
            + 0.15 * reverse_safety_score
        )
    else:
        reverse_score = _progress_lower(reverse_frac, floor=0.18, perfect=0.02)
    smoothness_score = 0.35 * _progress_lower(mean_action, floor=0.78, perfect=0.22) + 0.65 * _progress_lower(
        mean_du, floor=0.18, perfect=0.035
    )
    speed_safety_score = _progress_lower(max_trailer_speed, floor=1.50, perfect=0.55)
    safety_score = min(finite_score, workspace_score, no_go_score, hitch_safety_score, speed_safety_score)
    metric_validity_gate = min(finite_score, hitch_safety_score, workspace_score, no_go_score)
    docking_progress_gate = _progress_upper(progress_frac, floor=0.08, perfect=0.45)
    simultaneous_success = min(
        position_score,
        orientation_score,
        hold_score,
        hitch_safety_score,
        workspace_score,
        no_go_score,
    )
    ungated_score = (
        0.18 * position_score
        + 0.12 * orientation_score
        + 0.16 * progress_score
        + 0.12 * hold_score
        + 0.15 * hitch_safety_score
        + 0.10 * workspace_score
        + 0.07 * no_go_score
        + 0.06 * reverse_score
        + 0.04 * smoothness_score
    )
    achievement_signal = (
        0.30 * position_score
        + 0.18 * orientation_score
        + 0.18 * progress_score
        + 0.16 * hold_score
        + 0.12 * hitch_safety_score
        + 0.06 * no_go_score
    )
    achievement_gate = _progress_upper(achievement_signal, floor=0.20, perfect=0.72)
    score = ungated_score * achievement_gate
    if safety_score <= 0.0:
        score *= 0.15
    if not finite:
        score *= 0.10

    return {
        "id": scenario.get("id", "unknown"),
        "family": scenario.get("family", "unknown"),
        "score": _clamp01(score),
        "position": position_score * metric_validity_gate,
        "orientation": orientation_score * metric_validity_gate * docking_progress_gate,
        "progress": progress_score * metric_validity_gate,
        "hold": hold_score * metric_validity_gate * docking_progress_gate,
        "hitch_safety": hitch_safety_score,
        "workspace": workspace_score,
        "no_go": no_go_score,
        "reverse_control": reverse_score * metric_validity_gate,
        "smoothness": smoothness_score,
        "simultaneous_success": simultaneous_success,
        "finite": finite_score,
        "achievement_gate": achievement_gate,
        "achievement_signal": achievement_signal,
        "final_error": final_error,
        "yaw_error": yaw_error,
        "final_speed": final_speed,
        "progress_frac": progress_frac,
        "final_hitch_abs": final_hitch_abs,
        "max_hitch_abs_after_recovery": max_hitch_abs,
        "unsafe_hitch_frac": unsafe_hitch_frac,
        "min_workspace_margin": min_workspace_margin,
        "min_no_go_clearance": min_no_go_clearance,
        "reverse_frac": reverse_frac,
        "reverse_progress_frac": reverse_progress_m / max(initial_error, 1e-6),
        "reverse_safe_frac": reverse_safe_steps / max(1, reverse_steps),
        "mean_action": mean_action,
        "mean_du": mean_du,
        "error": error,
    }


def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | None,
    private: Path,
) -> dict[str, Any]:
    """Score a submitted docking policy on hidden deterministic scenarios."""
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
        scenario_results = []
        for scenario in scenarios:
            with PolicyWorker(policy_path, timeout_s=0.25, cwd=POLICY_CWD) as worker:
                scenario_results.append(_scenario_score(_PolicyCaller(worker), scenario))
    except Exception as exc:  # noqa: BLE001
        return {
            "score": 0.0,
            "subscores": {"policy_present": 1.0, "rollout_valid": 0.0},
            "weights": {"policy_present": 0.1, "rollout_valid": 0.9},
            "metadata": {"error": str(exc)},
        }

    scores = np.array([result["score"] for result in scenario_results], dtype=float)
    avg_score = float(np.mean(scores)) if len(scores) else 0.0
    worst_score = float(np.min(scores)) if len(scores) else 0.0
    subscore_keys = [
        "position",
        "orientation",
        "progress",
        "hold",
        "hitch_safety",
        "workspace",
        "no_go",
        "reverse_control",
        "smoothness",
    ]
    subscores = {
        key: float(np.mean([result[key] for result in scenario_results]))
        for key in subscore_keys
    }
    subscores["policy_present"] = 1.0
    subscores["worst_case"] = worst_score
    weights = {
        "policy_present": 0.0,
        "position": 0.130,
        "orientation": 0.065,
        "progress": 0.130,
        "hold": 0.045,
        "hitch_safety": 0.210,
        "workspace": 0.025,
        "no_go": 0.025,
        "reverse_control": 0.190,
        "smoothness": 0.060,
        "worst_case": 0.120,
    }
    raw_headline = _clamp01(sum(subscores[key] * weight for key, weight in weights.items()))
    headline = _calibrate_headline(raw_headline)
    rubric_rows = _rubric_rows(subscores, weights)

    return {
        "score": headline,
        "subscores": subscores,
        "weights": weights,
        "structured_subscores": rubric_rows,
        "metadata": {
            "num_scenarios": len(scenario_results),
            "raw_headline_score": raw_headline,
            "weighted_subscore_total": raw_headline,
            "acceptance_cutoff_unchanged_below": ACCEPTANCE_CUTOFF,
            "oracle_reference_raw_headline": ORACLE_RAW_HEADLINE,
            "calibration_note": "Scores at or below the acceptance cutoff are unchanged; the deterministic oracle raw headline is normalized to 1.0.",
            "avg_scenario_score": avg_score,
            "worst_scenario_score": worst_score,
            "scenario_details_redacted": True,
            "rubric_breakdown": rubric_rows,
            "diagnostic_gates": {
                "finite_mean": float(np.mean([result["finite"] for result in scenario_results])),
                "achievement_gate_mean": float(
                    np.mean([result["achievement_gate"] for result in scenario_results])
                ),
                "completion_gate_mean": float(
                    np.mean([result["simultaneous_success"] for result in scenario_results])
                ),
                "mean_reverse_fraction": float(np.mean([result["reverse_frac"] for result in scenario_results])),
                "mean_reverse_progress_fraction": float(
                    np.mean([result["reverse_progress_frac"] for result in scenario_results])
                ),
                "mean_reverse_safe_fraction": float(
                    np.mean([result["reverse_safe_frac"] for result in scenario_results])
                ),
            },
        },
    }
