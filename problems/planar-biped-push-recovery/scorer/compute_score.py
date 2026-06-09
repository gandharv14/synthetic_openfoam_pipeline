"""Deterministic grader for the planar biped push-recovery policy task.

The submitted ``policy.py`` is exercised against a fixed MuJoCo planar biped
under four hidden deterministic scenarios (quiet stand, lateral push from
each side, low-friction push) and probed for feedback sensitivity at two
out-of-equilibrium poses. Every rollout uses pinned timestep, integrator,
seed, initial state and friction so scores are reproducible bit-for-bit.

Anti-cheat posture:
  * The biped model is fixed at ``data/planar_biped.xml``; the agent cannot
    edit morphology, masses, contacts, or actuators.
  * The probe explicitly compares hip/ankle commands at +0.10 rad and
    -0.10 rad torso pitch, so policies that return a constant action are
    detected and fail ``feedback_sensitive``.
  * Stability and recovery checks bound torso height, pitch envelope,
    horizontal drift, joint velocity and final residual error so collapsing,
    drifting, or jittering policies fail even when they don't NaN.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from grading import PolicyWorker, RubricBuilder


# 200 Hz controller cadence — matches the rendering pipeline and gives the
# agent's policy ~25% of a 10ms wall budget per call inside the worker.
CONTROL_SKIP = 5
MAX_POLICY_STEP_SEC = 0.25


def _model_path(private: Path) -> Path:
    candidates = [
        Path("/data/planar_biped.xml"),
        private / "planar_biped.xml",
        Path(__file__).resolve().parents[1] / "data" / "planar_biped.xml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find planar_biped.xml")


def _cases_path(private: Path) -> Path:
    candidates = [
        private / "eval_cases.json",
        Path(__file__).resolve().parent / "data" / "eval_cases.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("could not find eval_cases.json")


def _build_obs(model: mujoco.MjModel, data: mujoco.MjData, step: int) -> dict[str, Any]:
    return {
        "time": float(data.time),
        "step": int(step),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "sensordata": data.sensordata.copy(),
        "ctrl": data.ctrl.copy(),
        "nu": int(model.nu),
        "nq": int(model.nq),
        "nv": int(model.nv),
    }


def _coerce_action(action: Any, model: mujoco.MjModel) -> np.ndarray:
    values = np.asarray(action, dtype=float).reshape(-1)
    if values.size != model.nu:
        raise ValueError(f"policy action size {values.size} does not match model.nu {model.nu}")
    if not np.isfinite(values).all():
        raise ValueError("policy action contains non-finite values")
    return np.clip(values, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])


def _make_model(model_path: Path, friction_scale: float = 1.0) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id >= 0:
        model.geom_friction[floor_id, 0] *= float(friction_scale)
    return model


def _rollout_case(model_path: Path, policy_path: Path, case: dict[str, Any]) -> dict[str, float | bool | str]:
    model = _make_model(model_path, friction_scale=float(case.get("friction_scale", 1.0)))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    q0 = np.asarray(case["initial_qpos"], dtype=float)
    data.qpos[: q0.size] = q0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    initial_x = float(data.xpos[torso_id, 0])
    metrics: dict[str, float | bool | str] = {
        "no_nan": True,
        "valid_actions": True,
        "min_z": float(data.xpos[torso_id, 2]),
        "max_abs_pitch": abs(float(data.qpos[2])),
        "max_abs_x": 0.0,
        "max_qvel_norm": 0.0,
        "final_abs_pitch": abs(float(data.qpos[2])),
        "final_abs_x": 0.0,
        "final_z": float(data.xpos[torso_id, 2]),
    }

    steps = int(float(case["duration"]) / model.opt.timestep)
    last_ctrl = np.zeros(model.nu)
    try:
        with PolicyWorker(policy_path, timeout_s=MAX_POLICY_STEP_SEC) as policy:
            for step in range(steps):
                t = step * model.opt.timestep
                data.xfrc_applied[:] = 0.0
                for push in case.get("pushes", []):
                    start = float(push["time"])
                    stop = start + float(push["duration"])
                    if start <= t < stop:
                        data.xfrc_applied[torso_id, 0] += float(push["force"])

                if step % CONTROL_SKIP == 0:
                    last_ctrl = _coerce_action(policy.act(_build_obs(model, data, step)), model)
                data.ctrl[:] = last_ctrl
                mujoco.mj_step(model, data)

                finite = np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                if not finite:
                    metrics["no_nan"] = False
                    break
                metrics["min_z"] = min(float(metrics["min_z"]), float(data.xpos[torso_id, 2]))
                metrics["max_abs_pitch"] = max(
                    float(metrics["max_abs_pitch"]), abs(float(data.qpos[2]))
                )
                metrics["max_abs_x"] = max(
                    float(metrics["max_abs_x"]), abs(float(data.xpos[torso_id, 0] - initial_x))
                )
                metrics["max_qvel_norm"] = max(
                    float(metrics["max_qvel_norm"]), float(np.linalg.norm(data.qvel))
                )
    except Exception as exc:  # noqa: BLE001 - policy failures are grader feedback.
        metrics["valid_actions"] = False
        metrics["no_nan"] = False
        metrics["error"] = str(exc)

    metrics["final_abs_pitch"] = abs(float(data.qpos[2]))
    metrics["final_abs_x"] = abs(float(data.xpos[torso_id, 0] - initial_x))
    metrics["final_z"] = float(data.xpos[torso_id, 2])
    return metrics


def _probe_policy(policy_path: Path, model_path: Path) -> dict[str, bool | float | str]:
    """Query the policy at three poses to detect feedback structure.

    Returns ``feedback_sensitive`` (the hip/ankle command differs between
    forward and backward lean by more than 0.05 rad — catches constant
    policies), ``stabilizing_sign`` (the delta has the sign that would
    counteract the lean, given this biped's actuator wiring), and
    ``valid`` (the policy responded with finite 6-vectors on all three
    poses).
    """
    model = _make_model(model_path)
    neutral = {
        "time": 0.0,
        "step": 0,
        "qpos": np.array([0.0, 0.0, 0.0, 0.08, -0.16, 0.08, 0.08, -0.16, 0.08]),
        "qvel": np.zeros(model.nv),
        "sensordata": np.zeros(model.nsensor),
        "ctrl": np.zeros(model.nu),
        "nu": int(model.nu),
        "nq": int(model.nq),
        "nv": int(model.nv),
    }
    lean_fwd = dict(neutral)
    lean_back = dict(neutral)
    lean_fwd["qpos"] = neutral["qpos"].copy()
    lean_back["qpos"] = neutral["qpos"].copy()
    lean_fwd["qvel"] = neutral["qvel"].copy()
    lean_back["qvel"] = neutral["qvel"].copy()
    lean_fwd["qpos"][2] = 0.10
    lean_back["qpos"][2] = -0.10

    try:
        with PolicyWorker(policy_path, timeout_s=MAX_POLICY_STEP_SEC) as policy:
            a0 = _coerce_action(policy.act(neutral), model)
            a_plus = _coerce_action(policy.act(lean_fwd), model)
            a_minus = _coerce_action(policy.act(lean_back), model)
    except Exception as exc:  # noqa: BLE001
        return {"valid": False, "feedback_sensitive": False, "stabilizing_sign": False, "error": str(exc)}

    hip_ankle = np.array([0, 2, 3, 5])  # left_hip, left_ankle, right_hip, right_ankle
    delta = float(np.mean(a_plus[hip_ankle]) - np.mean(a_minus[hip_ankle]))
    return {
        "valid": True,
        "feedback_delta": delta,
        "feedback_sensitive": abs(delta) > 0.05,
        "stabilizing_sign": delta > 0.0,
    }


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> dict[str, Any]:
    """Score a submitted biped feedback policy using deterministic rollouts."""
    policy_path = workspace / "policy.py"
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)

    try:
        model_path = _model_path(private)
        cases = json.loads(_cases_path(private).read_text())
        model = _make_model(model_path)
    except Exception as exc:
        rb.metadata["setup_error"] = str(exc)
        model_path = None
        cases = []
        model = None

    probe = {"valid": False, "feedback_sensitive": False, "stabilizing_sign": False}
    metrics_by_case: dict[str, dict[str, float | bool | str]] = {}
    if policy_path.exists() and model_path is not None:
        probe = _probe_policy(policy_path, model_path)
        for case in cases:
            metrics_by_case[str(case["name"])] = _rollout_case(model_path, policy_path, case)

    def case(name: str) -> dict[str, float | bool | str]:
        return metrics_by_case.get(name, {})

    # ── Structural / API criteria ─────────────────────────────────────────
    @rb.criterion(
        id="policy_file_exists",
        weight=0.5,
        description=(
            "Policy file is present at /tmp/output/policy.py. The grader cannot "
            "evaluate anything without this file, so this criterion is the "
            "minimum bar — present, importable Python at the canonical output path."
        ),
    )
    def _():
        return policy_path.exists()

    @rb.criterion(
        id="policy_action_valid",
        weight=0.8,
        description=(
            "Calling policy.act(obs) on a neutral standing observation returns "
            "a finite 6-element action (one target per position actuator). "
            "Catches submissions that import-fail, raise on the first call, "
            "return wrong-shaped output, or emit NaN/inf."
        ),
    )
    def _():
        return bool(probe.get("valid"))

    @rb.criterion(
        id="feedback_sensitive",
        weight=0.8,
        description=(
            "The policy's hip/ankle commands differ by more than 0.05 rad "
            "between a forward (+0.10 rad pitch) and backward (-0.10 rad pitch) "
            "torso lean observation. Any constant policy — including the "
            "all-zero baseline — has delta=0 and fails here."
        ),
    )
    def _():
        return bool(probe.get("feedback_sensitive"))

    @rb.criterion(
        id="stabilizing_feedback_sign",
        weight=0.8,
        description=(
            "Forward lean produces a more positive hip/ankle target than "
            "backward lean — the sign convention that actually rotates this "
            "biped's torso back toward vertical. A policy with reversed-sign "
            "feedback (destabilizing) passes 'feedback_sensitive' but fails here."
        ),
    )
    def _():
        return bool(probe.get("stabilizing_sign"))

    # ── Fixed-model sanity (anti-cheat anchor) ────────────────────────────
    @rb.criterion(
        id="fixed_model_sanity",
        weight=0.4,
        description=(
            "The hidden biped model still has the expected dimensions "
            "(nq=9, nv=9, nu=6). Anchors the contract that the agent's policy "
            "API must match — if this fails, the model itself was tampered "
            "with on the grading side and other failures are not actionable."
        ),
    )
    def _():
        return model is not None and model.nq == 9 and model.nv == 9 and model.nu == 6

    # ── Quiet-stand criteria (catch unresponsive / jittery policies) ──────
    @rb.criterion(
        id="nominal_standing_height",
        weight=1.0,
        description=(
            "In the no-push 'quiet stand' rollout, the torso height never "
            "drops below 1.03 m (vs. ~1.08 m at rest). Catches policies whose "
            "default action collapses the legs even before any disturbance."
        ),
    )
    def _():
        return float(case("nominal_stand").get("min_z", 0.0)) >= 1.03

    @rb.criterion(
        id="nominal_pose_bound",
        weight=1.0,
        description=(
            "In the quiet-stand rollout the torso pitch envelope stays within "
            "±0.02 rad (~1.15°) and horizontal drift within ±0.02 m. A policy "
            "that issues constant zero targets oscillates above this bound; "
            "a real feedback policy holds the stance still."
        ),
    )
    def _():
        m = case("nominal_stand")
        return float(m.get("max_abs_pitch", math.inf)) <= 0.02 and float(
            m.get("max_abs_x", math.inf)
        ) <= 0.02

    @rb.criterion(
        id="nominal_quietness",
        weight=0.6,
        description=(
            "Peak joint-velocity norm during the quiet stand stays under "
            "3.0 rad/s. Detects fundamentally jittery or oscillating "
            "controllers that nominally stand but pump energy into the joints; "
            "a converged feedback policy stays well under this bound."
        ),
    )
    def _():
        return float(case("nominal_stand").get("max_qvel_norm", math.inf)) <= 3.0

    # ── Lateral push survival/recovery (symmetric pair) ───────────────────
    @rb.criterion(
        id="left_push_survival",
        weight=1.2,
        description=(
            "Under a 42 N positive-x torso push lasting 0.12 s starting at "
            "t=0.7 s, the biped does not fall: torso height stays ≥1.02 m and "
            "peak |pitch| stays ≤0.18 rad (~10°). Distinguishes 'survived the "
            "impact' from 'returned to upright', graded separately below."
        ),
    )
    def _():
        m = case("left_push_recovery")
        return float(m.get("min_z", 0.0)) >= 1.02 and float(
            m.get("max_abs_pitch", math.inf)
        ) <= 0.18

    @rb.criterion(
        id="left_push_recovery",
        weight=1.2,
        description=(
            "After the positive-x push (same scenario as the survival "
            "criterion), the biped settles back near its initial pose: final "
            "|pitch| ≤0.04 rad and final horizontal drift ≤0.10 m at the end "
            "of the 3.5 s rollout. Captures the 'recover', not just 'survive'."
        ),
    )
    def _():
        m = case("left_push_recovery")
        return float(m.get("final_abs_pitch", math.inf)) <= 0.04 and float(
            m.get("final_abs_x", math.inf)
        ) <= 0.10

    @rb.criterion(
        id="right_push_survival",
        weight=1.2,
        description=(
            "Mirror of left_push_survival under a -42 N (negative-x) push. "
            "Same bounds (height ≥1.02 m, peak |pitch| ≤0.18 rad). Together "
            "with left_push_survival this catches asymmetric controllers that "
            "are tuned for only one push direction."
        ),
    )
    def _():
        m = case("right_push_recovery")
        return float(m.get("min_z", 0.0)) >= 1.02 and float(
            m.get("max_abs_pitch", math.inf)
        ) <= 0.18

    @rb.criterion(
        id="right_push_recovery",
        weight=1.2,
        description=(
            "Mirror of left_push_recovery: after the -42 N push the biped "
            "settles to final |pitch| ≤0.04 rad and final |x| drift ≤0.10 m. "
            "Pairing this with left_push_recovery prevents the policy from "
            "trading off recovery quality across sides."
        ),
    )
    def _():
        m = case("right_push_recovery")
        return float(m.get("final_abs_pitch", math.inf)) <= 0.04 and float(
            m.get("final_abs_x", math.inf)
        ) <= 0.10

    # ── Low-friction robustness (perturbation of contact parameters) ──────
    @rb.criterion(
        id="low_friction_robustness",
        weight=1.1,
        description=(
            "With floor friction reduced to 72% of nominal and a smaller 30 N "
            "push at t=0.9 s, the biped still stands (height ≥1.02 m), keeps "
            "|pitch| ≤0.07 rad, and limits horizontal drift to ≤0.10 m. "
            "Detects policies overfit to the nominal friction value."
        ),
    )
    def _():
        m = case("low_friction_small_push")
        return float(m.get("min_z", 0.0)) >= 1.02 and float(
            m.get("max_abs_pitch", math.inf)
        ) <= 0.07 and float(m.get("max_abs_x", math.inf)) <= 0.10

    # ── Global numerical / energy sanity ──────────────────────────────────
    @rb.criterion(
        id="all_rollouts_finite",
        weight=1.0,
        description=(
            "Across every hidden rollout the state stays finite (no NaN/inf) "
            "and the joint-velocity-norm peak stays ≤8.0 rad/s. Bounds away "
            "from physically implausible energy spikes that a policy could "
            "use to mask drift, and from solver blow-ups masquerading as "
            "successful rollouts."
        ),
    )
    def _():
        return bool(metrics_by_case) and all(
            bool(m.get("no_nan"))
            and bool(m.get("valid_actions"))
            and float(m.get("max_qvel_norm", math.inf)) <= 8.0
            for m in metrics_by_case.values()
        )

    rb.metadata["case_metrics"] = metrics_by_case
    rb.metadata["probe"] = probe
    if "error" in probe:
        rb.metadata["policy_probe_error"] = probe["error"]
    return rb.grade().to_dict()
