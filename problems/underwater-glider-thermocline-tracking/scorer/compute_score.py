"""Deterministic scorer for underwater glider thermocline tracking."""

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

from glider_env import (  # noqa: E402
    build_model,
    current_at,
    kinematic_step,
    observation,
    plume_clearance,
    reset_data,
    sample_reached,
    target_depth_at,
    workspace_margin,
)

ACCEPTANCE_CUTOFF = 0.40
ORACLE_RAW_HEADLINE = 0.8196059875602726

CRITERION_DESCRIPTIONS = {
    "ordered_samples": (
        "Safety-gated fraction of hidden thermocline sample windows scanned in order with continuous dwell; "
        "the headline completion gate is zero below the weakest hidden ordered-sample score 0.70 and full at 0.90."
    ),
    "thermocline_tracking": (
        "Safety-gated mean and p90 depth error while crossing the transect; full credit is near mean<=0.050 "
        "and p90<=0.095, with little credit by mean>=0.26 or p90>=0.34."
    ),
    "finish_quality": (
        "Safety-gated final-window position, depth, and drift near the requested finish after all samples; full "
        "credit is near distance<=0.070, depth error<=0.040, drift<=0.045, and the headline gate is full at "
        "weakest hidden finish score 0.50 after starting at 0.20."
    ),
    "safety": (
        "Workspace and plume clearance over the rollout; credit is full around >=0.045 m clearance with no unsafe "
        "steps and degrades toward zero by roughly -0.115 m clearance shortfall or 6% unsafe steps."
    ),
    "control_quality": (
        "Moderate buoyancy/pitch commands with limited chatter; full credit is near mean action<=0.28 and "
        "mean command delta<=0.055, with little credit by 1.05 and 0.55 respectively."
    ),
    "worst_case": "Worst hidden scenario aggregate score after safety gates across current and thermocline perturbations.",
    "policy_present": "Submitted /tmp/output/policy.py imports and exposes act(obs), get_action(obs), or Policy.act(obs).",
}


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


def _calibrate(raw_score: float) -> float:
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


class _PolicyCaller:
    # PolicyWorker instantiates module.Policy() when no module-level act()
    # exists, so calling "act" here supports both act(obs) and Policy.act(obs).
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


def _rubric_rows(subscores: dict[str, float], weights: dict[str, float]) -> list[dict[str, Any]]:
    rows = []
    for key, score in subscores.items():
        description = CRITERION_DESCRIPTIONS.get(key, key)
        rows.append(
            {
                "name": description,
                "label": description,
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


def _scenario_score(policy: _PolicyCaller, scenario: dict[str, Any]) -> dict[str, Any]:
    model = build_model(scenario)
    data = reset_data(model, scenario)
    dt = float(model.opt.timestep)
    duration = float(scenario.get("duration", 10.0))
    steps = int(duration / dt)
    sample_index = 0
    sample_hold_counter = 0
    samples = scenario.get("samples", [])
    sample_hold_time = float(scenario.get("sample_hold_time", 0.18))
    sample_hold_steps = max(1, int(math.ceil(sample_hold_time / dt)))
    finish = np.array(scenario["finish"], dtype=float)
    min_clearance = 10.0
    unsafe_steps = 0
    actions: list[np.ndarray] = []
    tracking_errors: list[float] = []
    final_window: list[tuple[float, float, float]] = []
    error: str | None = None

    for step_i in range(steps):
        time_sec = step_i * dt
        point = np.array(data.qpos[:2], dtype=float)
        if sample_index < len(samples) and sample_reached(point, scenario, samples[sample_index]):
            sample_hold_counter += 1
            if sample_hold_counter >= sample_hold_steps:
                sample_index += 1
                sample_hold_counter = 0
        elif sample_index < len(samples):
            sample_hold_counter = 0
        sample_hold_progress = sample_hold_counter / sample_hold_steps if sample_index < len(samples) else 1.0
        obs = observation(model, data, scenario, time_sec, sample_index, sample_hold_progress)
        try:
            action = np.array(policy(obs), dtype=float)
            clipped = kinematic_step(model, data, scenario, action, time_sec)
        except Exception as exc:  # noqa: BLE001
            error = str(exc)
            break
        actions.append(clipped)
        point = np.array(data.qpos[:2], dtype=float)
        clearance = min(
            workspace_margin(point, scenario.get("workspace")),
            plume_clearance(point, scenario),
        )
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            unsafe_steps += 1
        if point[0] > 0.18:
            tracking_errors.append(abs(float(point[1]) - target_depth_at(scenario, float(point[0]))))
        if time_sec > duration - 1.25:
            current = current_at(scenario, point, time_sec)
            drift_relative_speed = float(np.linalg.norm(np.array(data.qvel[:2], dtype=float) + current))
            final_window.append((float(np.linalg.norm(point - finish)), abs(float(point[1]) - finish[1]), drift_relative_speed))
        if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
            error = "non-finite simulation state"
            break

    final_point = np.array(data.qpos[:2], dtype=float)
    final_dist = float(np.linalg.norm(final_point - finish))
    sample_partial = sample_hold_counter / sample_hold_steps if sample_index < len(samples) else 0.0
    sample_progress = (sample_index + sample_partial) / max(1, len(samples))
    sample_gate = _progress_upper(sample_progress, 0.25, 1.0)
    mean_track_error = float(np.mean(tracking_errors)) if tracking_errors else 1.0
    p90_track_error = float(np.percentile(tracking_errors, 90)) if tracking_errors else 1.0
    tracking = _clamp01(
        0.62 * _progress_lower(mean_track_error, 0.26, 0.050)
        + 0.38 * _progress_lower(p90_track_error, 0.34, 0.095)
    )
    finish_unlocked = 1.0 if sample_index == len(samples) else 0.0
    if final_window:
        mean_final_dist = float(np.mean([item[0] for item in final_window]))
        mean_final_depth = float(np.mean([item[1] for item in final_window]))
        mean_final_speed = float(np.mean([item[2] for item in final_window]))
    else:
        mean_final_dist = final_dist
        mean_final_depth = abs(float(final_point[1]) - finish[1])
        mean_final_speed = float(np.linalg.norm(data.qvel[:2]))
    finish_quality = finish_unlocked * _clamp01(
        0.55 * _progress_lower(mean_final_dist, 0.35, 0.070)
        + 0.25 * _progress_lower(mean_final_depth, 0.22, 0.040)
        + 0.20 * _progress_lower(mean_final_speed, 0.28, 0.045)
    )
    unsafe_fraction = unsafe_steps / max(1, len(actions))
    clearance_score = _progress_lower(max(0.0, 0.045 - min_clearance), 0.16, 0.0)
    safety = _clamp01(clearance_score * _progress_lower(unsafe_fraction, 0.06, 0.0))
    if actions:
        arr = np.vstack(actions)
        mean_action = float(np.mean(np.linalg.norm(arr, axis=1)))
        mean_du = float(np.mean(np.linalg.norm(np.diff(arr, axis=0), axis=1))) if len(arr) > 1 else 0.0
    else:
        mean_action = 1.0
        mean_du = 1.0
    control_quality = _clamp01(
        0.55 * _progress_lower(mean_action, 1.05, 0.28)
        + 0.45 * _progress_lower(mean_du, 0.55, 0.055)
    )

    safety_gate = _progress_upper(safety, 0.20, 0.78)
    ordered_samples = sample_progress * safety_gate
    tracking *= safety_gate
    finish_quality *= safety_gate
    scenario_score = _clamp01(
        0.34 * ordered_samples
        + 0.24 * tracking
        + 0.18 * finish_quality
        + 0.16 * safety
        + 0.08 * control_quality
    )
    return {
        "score": scenario_score if error is None else min(scenario_score, 0.25),
        "ordered_samples": ordered_samples,
        "thermocline_tracking": tracking,
        "finish_quality": finish_quality,
        "safety": safety,
        "control_quality": control_quality,
        "samples_reached": sample_index,
        "num_samples": len(samples),
        "final_dist": final_dist,
        "mean_track_error": mean_track_error,
        "p90_track_error": p90_track_error,
        "min_clearance": min_clearance,
        "unsafe_fraction": unsafe_fraction,
        "mean_action": mean_action,
        "mean_du": mean_du,
        "error": error,
    }


def compute_score(
    workspace: Path,
    trajectory: list[dict[str, Any]] | None,
    private: Path,
) -> dict[str, Any]:
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
            with PolicyWorker(policy_path, timeout_s=0.20, cwd=POLICY_CWD) as worker:
                scenario_results.append(_scenario_score(_PolicyCaller(worker), scenario))
    except Exception as exc:  # noqa: BLE001
        return {
            "score": 0.0,
            "subscores": {"policy_present": 1.0, "rollout_valid": 0.0},
            "weights": {"policy_present": 0.1, "rollout_valid": 0.9},
            "metadata": {"error": str(exc)},
        }

    weights = {
        "ordered_samples": 0.24,
        "thermocline_tracking": 0.22,
        "finish_quality": 0.20,
        "safety": 0.14,
        "control_quality": 0.10,
        "worst_case": 0.10,
        "policy_present": 0.0,
    }
    scores = np.array([item["score"] for item in scenario_results], dtype=float)
    worst_scenario_score = float(np.min(scores)) if len(scores) else 0.0
    min_ordered_samples = float(np.min([item["ordered_samples"] for item in scenario_results])) if scenario_results else 0.0
    min_finish_quality = float(np.min([item["finish_quality"] for item in scenario_results])) if scenario_results else 0.0
    completion_gate = _progress_upper(min_ordered_samples, 0.70, 0.90) * _progress_upper(min_finish_quality, 0.20, 0.50)
    subscores = {
        "ordered_samples": float(np.mean([item["ordered_samples"] for item in scenario_results])),
        "thermocline_tracking": float(np.mean([item["thermocline_tracking"] for item in scenario_results])),
        "finish_quality": float(np.mean([item["finish_quality"] for item in scenario_results])),
        "safety": float(np.mean([item["safety"] for item in scenario_results])),
        "control_quality": float(np.mean([item["control_quality"] for item in scenario_results])),
        "worst_case": worst_scenario_score,
        "policy_present": 1.0,
    }
    base_weighted_total = sum(subscores[key] * weight for key, weight in weights.items())
    raw = base_weighted_total * completion_gate
    headline = _calibrate(raw)
    rows = _rubric_rows(subscores, weights)
    return {
        "score": headline,
        "subscores": subscores,
        "weights": weights,
        "structured_subscores": rows,
        "scoring_mode": "weighted",
        "metadata": {
            "return_shape": "rubric_grade",
            "acceptance_cutoff_unchanged_below": ACCEPTANCE_CUTOFF,
            "oracle_reference_raw_headline": ORACLE_RAW_HEADLINE,
            "base_weighted_total_before_completion_gate": base_weighted_total,
            "min_ordered_samples": min_ordered_samples,
            "min_finish_quality": min_finish_quality,
            "completion_gate": completion_gate,
            "raw_headline_score": raw,
            "reported_final_score": headline,
            "avg_scenario_score": float(np.mean(scores)) if len(scores) else 0.0,
            "worst_scenario_score": worst_scenario_score,
            "num_scenarios": len(scenario_results),
            "scenario_details_redacted": True,
            "rubric_breakdown": [
                {
                    "id": row["id"],
                    "criterion_id": row["criterion_id"],
                    "criterion": row["id"],
                    "description": row["description"],
                    "label": row["label"],
                    "score": row["score"],
                    "weight": row["weight"],
                    "passed": row["score"] >= 0.5,
                    "reasoning": "",
                    "grading_type": "continuous",
                    "expected": row["description"],
                    "actual": None,
                }
                for row in rows
            ],
        },
    }
