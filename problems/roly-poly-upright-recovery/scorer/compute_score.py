"""Deterministic scorer for the roly-poly upright recovery task."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from grading import PolicyWorker, RubricBuilder, helpers  # noqa: F401

_SCORER_DIR = Path(__file__).resolve().parent
_TASK_DIR = _SCORER_DIR.parent
DATA_DIRS = [_TASK_DIR / "data", _SCORER_DIR / "data", Path("/data")]
for data_dir in DATA_DIRS:
    if data_dir.exists() and str(data_dir) not in sys.path:
        sys.path.insert(0, str(data_dir))

from roly_env import (  # noqa: E402
    BASE_BODY,
    CROWN_BODY,
    PIVOT_SITE,
    TILT_JOINT,
    load_model,
    run_rollout,
    subtree_bodies,
)


def _clamp01(v: float) -> float:
    return float(max(0.0, min(1.0, v)))


def _progress_lower(value: float, bad: float, good: float) -> float:
    if bad <= good:
        return 0.0
    return _clamp01((bad - value) / (bad - good))


def _progress_upper(value: float, floor: float, perfect: float) -> float:
    if perfect <= floor:
        return 0.0
    return _clamp01((value - floor) / (perfect - floor))


def _com_below_pivot(model: mujoco.MjModel) -> bool:
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    pivot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PIVOT_SITE)
    doll_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "doll")
    if pivot_id < 0 or doll_id < 0:
        return False
    pivot_z = float(data.site_xpos[pivot_id][2])
    total = 0.0
    com_z = 0.0
    for bid in subtree_bodies(model, doll_id):
        mass = float(model.body_mass[bid])
        if mass <= 0.0:
            continue
        total += mass
        com_z += mass * float(data.xipos[bid][2])
    if total <= 0.0:
        return False
    return (com_z / total) < pivot_z - 0.008


def _scenario_score(result: dict[str, Any], anchors: dict[str, Any]) -> float:
    if not result.get("finite", False) or not result.get("recovered", False):
        return 0.0
    upright = _progress_upper(
        float(result.get("hold_upright_z", 0.0)),
        anchors["hold_upright_floor"],
        anchors["hold_upright_perfect"],
    )
    tilt = _progress_lower(
        float(result.get("hold_tilt_abs", 1.0)),
        anchors["hold_tilt_floor"],
        anchors["hold_tilt_perfect"],
    )
    vel = _progress_lower(
        float(result.get("hold_tilt_vel", 1.0)),
        anchors["hold_vel_floor"],
        anchors["hold_vel_perfect"],
    )
    effort = float(result.get("effort", 0.0))
    if effort < float(anchors["effort_min_active"]):
        return 0.0
    effort_score = _progress_lower(
        effort,
        anchors["effort_floor"],
        anchors["effort_perfect"],
    )
    jerk = _progress_lower(
        float(result.get("jerk", 1.0)),
        anchors["jerk_floor"],
        anchors["jerk_perfect"],
    )
    return float(min(upright, tilt, vel, effort_score, jerk))


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> dict[str, Any]:
    _ = trajectory
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)
    anchors = json.loads((private / "anchors.json").read_text())
    scenarios = json.loads((private / "hidden_scenarios.json").read_text())

    xml_path = workspace / "model.xml"
    policy_path = workspace / "policy.py"
    model: mujoco.MjModel | None = None
    scenario_results: list[dict[str, Any]] = []

    if xml_path.exists():
        try:
            model = load_model(xml_path)
        except Exception as exc:  # noqa: BLE001
            rb.metadata["compile_error"] = str(exc)

    structure_ok = False
    if model is not None:
        has_tilt = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, TILT_JOINT) >= 0
        has_base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY) >= 0
        has_crown = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CROWN_BODY) >= 0
        has_pivot = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, PIVOT_SITE) >= 0
        sensors_ok = all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, s) >= 0
            for s in ("tilt_pos", "tilt_vel", "upright_axis")
        )
        ctrl_ok = False
        if model.nu == 1:
            lo, hi = model.actuator_ctrlrange[0]
            ctrl_ok = abs(float(lo)) <= 12.0 and abs(float(hi)) <= 12.0
        base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        crown_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CROWN_BODY)
        mass_ok = False
        if base_id >= 0 and crown_id >= 0:
            bm = float(model.body_mass[base_id])
            cm = float(model.body_mass[crown_id])
            mass_ok = bm >= 3.0 * cm and bm >= 0.8
        structure_ok = (
            has_tilt
            and has_base
            and has_crown
            and has_pivot
            and sensors_ok
            and model.nu == 1
            and ctrl_ok
            and int(model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4)
            and float(model.opt.timestep) <= 0.005
            and mass_ok
            and _com_below_pivot(model)
        )

        rollout_ok = structure_ok and policy_path.exists()
        if rollout_ok:
            with PolicyWorker(policy_path, timeout_s=3.0) as worker:
                for scenario in scenarios:
                    sid = scenario.get("id", "unknown")
                    try:
                        result = run_rollout(model, worker, scenario)
                        result["id"] = sid
                        result["score"] = _scenario_score(result, anchors)
                    except Exception as exc:  # noqa: BLE001
                        result = {
                            "id": sid,
                            "score": 0.0,
                            "finite": False,
                            "error": str(exc),
                        }
                    scenario_results.append(result)

    scored_rollouts = structure_ok and bool(scenario_results)
    completions = [float(r["score"]) for r in scenario_results]
    mean_completion = float(np.mean(completions)) if scored_rollouts else 0.0
    worst_completion = float(min(completions)) if scored_rollouts else 0.0

    @rb.criterion(id="compiled", weight=0.05, description="MJCF compiles")
    def _compiled():
        return model is not None

    @rb.criterion(
        id="structure",
        weight=0.10,
        description="tilt hinge, weighted base/crown, COM below pivot, sensors, RK4",
    )
    def _structure():
        return structure_ok

    @rb.criterion(
        id="task_completion",
        weight=0.20,
        description="Mean per-scenario upright recovery completion",
    )
    def _task_completion():
        return mean_completion if scored_rollouts else 0.0

    @rb.criterion(
        id="scenario_coverage",
        weight=0.65,
        description="Worst hidden-scenario upright recovery score",
    )
    def _scenario_coverage():
        return worst_completion if scored_rollouts else 0.0

    rb.metadata["scenario_scores"] = [{"id": r["id"], "score": r["score"]} for r in scenario_results]
    rb.metadata["worst_task_completion"] = worst_completion
    rb.metadata["mean_task_completion"] = mean_completion
    return rb.grade().to_dict()
