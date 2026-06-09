"""Deterministic scorer for GPU Morphing Wing Gust Rejection.

Submitted policies are isolated behind ``grading.PolicyWorker``. Hidden gust,
dropout, stiffness, payload, and actuator-fatigue schedules remain in the
grader process; the policy only receives public live state and target
observations.
"""

from __future__ import annotations

import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from grading import PolicyWorker as _BasePolicyWorker
from grading import RubricBuilder
from grading.policy_runner import _WORKER_SOURCE

MODEL_CANDIDATES = (
    Path("/data/morphing_wing.xml"),
    Path(__file__).resolve().parents[1] / "data" / "morphing_wing.xml",
)

LEFT_TIP_SITE = "left_tip_site"
RIGHT_TIP_SITE = "right_tip_site"
LEFT_BODIES = ("left_inner", "left_mid", "left_tip")
RIGHT_BODIES = ("right_inner", "right_mid", "right_tip")
CONTROL_SKIP = 2
POLICY_TIMEOUT_SEC = 0.35
POLICY_WORKER_UID = 65534
POLICY_WORKER_GID = 65534
RECOVERY_TIP_ERROR_THRESHOLD = 0.105
RECOVERY_COVERAGE_TIME = 0.62
_WORKER_ENV_ALLOWLIST = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "MKL_NUM_THREADS",
        "MUJOCO_GL",
        "NVIDIA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PATH",
        "PYOPENGL_PLATFORM",
        "PYTHONHASHSEED",
        "TMP",
        "TMPDIR",
    }
)
ORACLE_CALIBRATION_SUMMARY = {
    "joint_envelope_error": 0.151,
    "tip_tail_error": 0.156,
    "camber_envelope_error": 0.063,
    "fault_recovered_fraction": 0.528,
    "mean_jitter": 0.0044,
    "max_qvel": 50.71,
    "mean_effort": 0.388,
    "peak_command": 0.985,
    "sat_fraction": 0.142,
}


class SandboxedPolicyWorker(_BasePolicyWorker):
    """Task-local policy runner that drops root before executing policy.py."""

    def _sandbox_user_kwargs(self) -> dict[str, Any]:
        if os.name != "posix" or not hasattr(os, "geteuid") or os.geteuid() != 0:
            return {}
        return {
            "user": POLICY_WORKER_UID,
            "group": POLICY_WORKER_GID,
            "extra_groups": [],
        }

    def _prepare_sandbox_access(self, sandbox_kwargs: dict[str, Any]) -> None:
        if not sandbox_kwargs:
            return
        try:
            policy_path = self.policy_path.resolve()
            tmp_root = Path(tempfile.gettempdir()).resolve()
        except OSError:
            return
        if policy_path == tmp_root or tmp_root not in policy_path.parents:
            return
        for directory in (policy_path.parent, *policy_path.parent.parents):
            if directory == tmp_root:
                break
            try:
                directory.chmod(directory.stat().st_mode | 0o755)
            except OSError:
                return
        for root, dirnames, filenames in os.walk(policy_path.parent, followlinks=False):
            root_path = Path(root)
            dirnames[:] = [name for name in dirnames if not (root_path / name).is_symlink()]
            for name in dirnames:
                try:
                    directory = root_path / name
                    directory.chmod(directory.stat().st_mode | 0o755)
                except OSError:
                    continue
            for name in filenames:
                file_path = root_path / name
                if file_path.is_symlink():
                    continue
                try:
                    stat_result = file_path.stat()
                    if stat_result.st_nlink == 1:
                        file_path.chmod(stat_result.st_mode | 0o444)
                except OSError:
                    continue

    @staticmethod
    def _worker_env() -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if key in _WORKER_ENV_ALLOWLIST
        }
        tmp_dir = tempfile.gettempdir()
        env["HOME"] = tmp_dir
        env.setdefault("TMPDIR", tmp_dir)
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        return env

    def start(self) -> None:
        if self._proc is not None:
            return
        if not self.policy_path.exists():
            raise FileNotFoundError(f"missing policy file: {self.policy_path}")
        self._stdout = queue.Queue()
        self._stderr_parts = []
        sandbox_kwargs = self._sandbox_user_kwargs()
        self._prepare_sandbox_access(sandbox_kwargs)
        self._proc = subprocess.Popen(
            [sys.executable, "-u", "-c", _WORKER_SOURCE, str(self.policy_path)],
            cwd=self.cwd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=self._worker_env(),
            **sandbox_kwargs,
        )
        assert self._proc.stdout is not None
        assert self._proc.stderr is not None
        self._stdout_thread = threading.Thread(
            target=self._drain_stdout, args=(self._proc.stdout,), daemon=True
        )
        self._stderr_thread = threading.Thread(
            target=self._drain_stderr, args=(self._proc.stderr,), daemon=True
        )
        self._stdout_thread.start()
        self._stderr_thread.start()


def _normalize_case(case: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(case)
    normalized.setdefault("dropouts", [])
    normalized.setdefault("gusts", [])
    return normalized


def _evaluation_cases(private: Path) -> tuple[dict[str, Any], ...]:
    hidden_cases_path = private / "hidden_cases.json"
    if not hidden_cases_path.exists():
        raise FileNotFoundError(
            f"hidden evaluation cases are required at {hidden_cases_path}"
        )
    return tuple(_normalize_case(case) for case in json.loads(hidden_cases_path.read_text()))


def _clamp01(value: float) -> float:
    if not math.isfinite(float(value)):
        return 0.0
    return float(max(0.0, min(1.0, value)))


def _lower_better(value: float, zero: float, full: float) -> float:
    if value <= full:
        return 1.0
    if value >= zero:
        return 0.0
    return _clamp01((zero - value) / (zero - full))


def _upper_better(value: float, zero: float, full: float) -> float:
    if value >= full:
        return 1.0
    if value <= zero:
        return 0.0
    return _clamp01((value - zero) / (full - zero))


def _model_path() -> Path:
    for path in MODEL_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("morphing_wing.xml not found")


def _target(case: dict[str, Any], t: float) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(case["base"], dtype=float)
    amp = np.asarray(case["amplitude"], dtype=float)
    phase = np.asarray(case["phase"], dtype=float)
    omega = 2.0 * math.pi * float(case["frequency"])
    arg = omega * float(t) + phase
    q = base + amp * np.sin(arg)
    qd = amp * omega * np.cos(arg)
    return q, qd


def _site_ids(model: mujoco.MjModel) -> tuple[int, int]:
    return (
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, LEFT_TIP_SITE),
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_TIP_SITE),
    )


def _tip_positions(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    fk_data: mujoco.MjData | None = None,
    site_ids: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    data = fk_data if fk_data is not None else mujoco.MjData(model)
    data.qpos[:] = np.asarray(qpos, dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    left_id, right_id = site_ids if site_ids is not None else _site_ids(model)
    return data.site_xpos[left_id].copy(), data.site_xpos[right_id].copy()


def _scale_branch(model: mujoco.MjModel, names: tuple[str, ...], scale: float) -> None:
    for name in names:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id >= 0:
            model.body_mass[body_id] *= scale
            model.body_inertia[body_id] *= scale


def _case_model(case: dict[str, Any]) -> mujoco.MjModel:
    model = mujoco.MjModel.from_xml_path(str(_model_path()))
    _scale_branch(model, LEFT_BODIES, float(case.get("left_payload_scale", 1.0)))
    _scale_branch(model, RIGHT_BODIES, float(case.get("right_payload_scale", 1.0)))
    model.dof_damping[:] *= float(case.get("damping_scale", 1.0))
    model.jnt_stiffness[:] *= float(case.get("stiffness_scale", 1.0))
    return model


def _obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    case: dict[str, Any],
    step: int,
    last_ctrl: np.ndarray,
    fk_data: mujoco.MjData,
    site_ids: tuple[int, int],
) -> dict[str, Any]:
    target_q, target_qd = _target(case, float(data.time))
    target_left, target_right = _tip_positions(model, target_q, fk_data, site_ids)
    left_id, right_id = site_ids
    return {
        "time": float(data.time),
        "step": int(step),
        "qpos": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "left_tip_pos": data.site_xpos[left_id].copy(),
        "right_tip_pos": data.site_xpos[right_id].copy(),
        "target_left_tip_pos": target_left,
        "target_right_tip_pos": target_right,
        "target_root_roll": float(target_q[0]),
        "target_left_camber": float(np.mean(target_q[1:4])),
        "target_right_camber": float(np.mean(target_q[4:7])),
        "last_ctrl": last_ctrl.copy(),
        "phase": float((float(data.time) * float(case["frequency"])) % 1.0),
    }


def _coerce_action(raw: Any, nu: int) -> tuple[np.ndarray, bool]:
    try:
        action = np.asarray(raw, dtype=float).reshape(-1)
    except Exception:  # noqa: BLE001 - submitted policy boundary
        return np.zeros(nu), False
    if action.size != nu or not np.isfinite(action).all():
        return np.zeros(nu), False
    clipped = np.clip(action, -1.0, 1.0)
    return clipped, bool(np.allclose(action, clipped, atol=1e-9))


def _dynamic_gain(case: dict[str, Any], t: float, nu: int) -> np.ndarray:
    raw_gains = np.asarray(case.get("actuator_gains", [1.0] * nu), dtype=float).reshape(-1)
    gains = np.ones(nu, dtype=float)
    count = min(nu, raw_gains.size)
    if count:
        gains[:count] = np.nan_to_num(raw_gains[:count], nan=1.0, posinf=1.0, neginf=1.0)
    for dropout in case.get("dropouts", []):
        start = float(dropout["start"])
        stop = start + float(dropout["duration"])
        joint = int(dropout["joint"])
        if start <= t < stop and 0 <= joint < nu:
            gains[joint] *= float(dropout.get("gain", 0.0))
    return gains[:nu]


def _apply_gusts(model: mujoco.MjModel, data: mujoco.MjData, case: dict[str, Any]) -> None:
    data.qfrc_applied[:] = 0.0
    t = float(data.time)
    for gust in case.get("gusts", []):
        start = float(gust["time"])
        duration = float(gust.get("duration", 0.05))
        if start <= t < start + duration:
            data.qfrc_applied[int(gust["joint"])] += float(gust["impulse"]) / max(duration, model.opt.timestep)


def _recover_time(
    times: np.ndarray,
    errors: np.ndarray,
    event_time: float,
    threshold: float,
    horizon: float = 0.95,
) -> float:
    mask = (times >= event_time + 0.08) & (times <= event_time + horizon)
    idxs = np.flatnonzero(mask)
    if idxs.size == 0:
        return horizon
    for idx in idxs:
        if errors[idx] <= threshold:
            return float(times[idx] - event_time)
    return horizon


def _rollout_case(policy_path: Path, case: dict[str, Any]) -> dict[str, Any]:
    model = _case_model(case)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    q0, _ = _target(case, 0.0)
    data.qpos[:] = q0 + np.asarray(case.get("initial_offset", [0.0] * 7), dtype=float)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    site_ids = _site_ids(model)
    fk_data = mujoco.MjData(model)

    steps = int(round(float(case["duration"]) / model.opt.timestep))
    last_ctrl = np.zeros(model.nu)
    actions: list[np.ndarray] = []
    joint_errors: list[float] = []
    tip_errors: list[float] = []
    roll_errors: list[float] = []
    camber_errors: list[float] = []
    camber_balance_errors: list[float] = []
    qvel_norms: list[float] = []
    times: list[float] = []
    valid_action_count = 0
    action_calls = 0
    finite = True
    action_contract = True
    error = ""

    try:
        with SandboxedPolicyWorker(
            policy_path,
            timeout_s=POLICY_TIMEOUT_SEC,
            cwd=policy_path.parent,
        ) as worker:
            for step in range(steps):
                if step % CONTROL_SKIP == 0:
                    action_calls += 1
                    raw = worker.act(_obs(model, data, case, step, last_ctrl, fk_data, site_ids))
                    last_ctrl, ok = _coerce_action(raw, model.nu)
                    action_contract = action_contract and ok
                    valid_action_count += int(ok)
                    actions.append(last_ctrl.copy())

                _apply_gusts(model, data, case)
                gains = _dynamic_gain(case, float(data.time), model.nu)
                data.ctrl[:] = np.clip(last_ctrl * gains, -1.0, 1.0)
                mujoco.mj_step(model, data)

                if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
                    finite = False
                    break

                sample_time = float(data.time)
                target_q, _target_qd = _target(case, sample_time)
                mujoco.mj_forward(model, data)
                target_left, target_right = _tip_positions(model, target_q, fk_data, site_ids)
                left_id, right_id = site_ids
                left_error = np.linalg.norm(data.site_xpos[left_id] - target_left)
                right_error = np.linalg.norm(data.site_xpos[right_id] - target_right)
                tip_errors.append(float(0.5 * (left_error + right_error)))
                roll_errors.append(float(abs(data.qpos[0] - target_q[0])))
                left_delta = data.qpos[1:4] - target_q[1:4]
                right_delta = data.qpos[4:7] - target_q[4:7]
                left_camber = np.mean(left_delta)
                right_camber = np.mean(right_delta)
                internal_shape = np.concatenate([
                    left_delta - left_camber,
                    right_delta - right_camber,
                ])
                joint_errors.append(float(np.linalg.norm(internal_shape) / math.sqrt(6.0)))
                camber_errors.append(float(0.5 * (abs(left_camber) + abs(right_camber))))
                camber_balance_errors.append(float(abs(left_camber - right_camber)))
                qvel_norms.append(float(np.linalg.norm(data.qvel)))
                times.append(float(data.time))
    except Exception as exc:  # noqa: BLE001 - submitted policy boundary
        finite = False
        action_contract = False
        error = f"{type(exc).__name__}: {exc}"

    if not joint_errors:
        return {
            "id": case.get("id", "unknown"),
            "finite": False,
            "action_contract": False,
            "valid_action_fraction": 0.0,
            "mean_joint_error": 999.0,
            "p90_joint_error": 999.0,
            "mean_tip_error": 999.0,
            "p90_tip_error": 999.0,
            "final_tip_error": 999.0,
            "roll_error": 999.0,
            "camber_error": 999.0,
            "camber_balance_error": 999.0,
            "max_qvel": 999.0,
            "mean_effort": 0.0,
            "p95_effort": 0.0,
            "peak_command": 0.0,
            "mean_jitter": 999.0,
            "sat_fraction": 1.0,
            "recovery_time": 0.95,
            "fault_recovered": 0.0,
            "error": error,
        }

    joint = np.asarray(joint_errors)
    tip = np.asarray(tip_errors)
    roll = np.asarray(roll_errors)
    camber = np.asarray(camber_errors)
    camber_balance = np.asarray(camber_balance_errors)
    times_arr = np.asarray(times)
    acts = np.asarray(actions)
    event_times = [float(d["start"]) for d in case.get("dropouts", [])]
    event_times += [float(g["time"]) for g in case.get("gusts", [])]
    recoveries = [
        _recover_time(times_arr, tip, t, threshold=RECOVERY_TIP_ERROR_THRESHOLD)
        for t in event_times
    ]
    recovery_time = float(np.mean(recoveries)) if recoveries else 0.0
    fault_recovered = (
        float(np.mean([r <= RECOVERY_COVERAGE_TIME for r in recoveries]))
        if recoveries
        else 1.0
    )
    final_mask = times_arr >= (float(case["duration"]) - 0.80)
    final_tip = float(np.mean(tip[final_mask])) if np.any(final_mask) else float(tip[-1])
    deltas = np.diff(acts, axis=0) if acts.shape[0] > 1 else np.zeros((1, model.nu))

    effort_norm = np.linalg.norm(acts, axis=1) / math.sqrt(model.nu)
    return {
        "id": case.get("id", "unknown"),
        "finite": bool(finite),
        "action_contract": bool(action_contract),
        "valid_action_fraction": float(valid_action_count / max(1, action_calls)),
        "mean_joint_error": float(np.mean(joint)),
        "p90_joint_error": float(np.quantile(joint, 0.90)),
        "mean_tip_error": float(np.mean(tip)),
        "p90_tip_error": float(np.quantile(tip, 0.90)),
        "final_tip_error": final_tip,
        "roll_error": float(np.mean(roll)),
        "camber_error": float(np.mean(camber)),
        "camber_balance_error": float(np.mean(camber_balance)),
        "max_qvel": float(max(qvel_norms)),
        "mean_effort": float(np.mean(effort_norm)),
        "p95_effort": float(np.quantile(effort_norm, 0.95)),
        "peak_command": float(np.max(np.abs(acts))),
        "mean_jitter": float(np.mean(np.linalg.norm(deltas, axis=1) / math.sqrt(model.nu))),
        "sat_fraction": float(np.mean(np.abs(acts) > 0.96)),
        "recovery_time": recovery_time,
        "fault_recovered": fault_recovered,
        "error": error,
    }


def compute_score(
    workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path
) -> dict[str, Any]:
    rb = RubricBuilder(workspace=workspace, trajectory=trajectory, private=private)
    policy_path = workspace / "policy.py"
    cases = _evaluation_cases(private)
    model_ok = False
    model_contract_score = 0.0
    results: list[dict[str, Any]] = []
    setup_error = ""

    try:
        model = mujoco.MjModel.from_xml_path(str(_model_path()))
        model_ok = model.nq == 7 and model.nu == 7 and model.nsensor >= 16
        left_site, right_site = _site_ids(model)
        named_bodies_ok = all(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) >= 0
            for name in (*LEFT_BODIES, *RIGHT_BODIES)
        )
        model_contract_score = float(
            model_ok
            and left_site >= 0
            and right_site >= 0
            and named_bodies_ok
            and math.isclose(float(model.opt.timestep), 0.004, rel_tol=0.0, abs_tol=1e-12)
        )
    except Exception as exc:  # noqa: BLE001
        setup_error = str(exc)

    if not policy_path.exists():
        setup_error = "policy.py missing from workspace; no submitted policy was available for rollout"
    elif not model_ok and not setup_error:
        setup_error = "morphing_wing.xml did not match the expected nq=7, nu=7, nsensor>=16 contract"
    elif model_ok:
        for case in cases:
            results.append(_rollout_case(policy_path, case))

    failure_defaults = {
        "mean_effort": 0.0,
        "p95_effort": 0.0,
        "peak_command": 0.0,
        "sat_fraction": 1.0,
        "recovery_time": 0.95,
        "fault_recovered": 0.0,
    }

    def values(name: str) -> list[float]:
        if not results:
            return [failure_defaults.get(name, 999.0)]
        return [float(row[name]) for row in results]

    finite_fraction = float(np.mean([bool(row.get("finite", False)) for row in results])) if results else 0.0
    action_fraction = float(np.mean([row.get("valid_action_fraction", 0.0) for row in results])) if results else 0.0
    rollout_validity_score = min(finite_fraction, action_fraction)
    mean_joint = float(np.mean(values("mean_joint_error")))
    p90_joint = float(np.mean(values("p90_joint_error")))
    worst_joint = float(np.max(values("p90_joint_error")))
    mean_tip = float(np.mean(values("mean_tip_error")))
    p90_tip = float(np.mean(values("p90_tip_error")))
    worst_tip = float(np.max(values("p90_tip_error")))
    final_tip = float(np.mean(values("final_tip_error")))
    roll_error = float(np.mean(values("roll_error")))
    camber_error = float(np.mean(values("camber_error")))
    camber_balance = float(np.mean(values("camber_balance_error")))
    recovery = float(np.mean(values("recovery_time")))
    worst_recovery = float(np.max(values("recovery_time")))
    fault_recovered = float(np.mean(values("fault_recovered"))) if results else 0.0
    max_qvel = float(np.max(values("max_qvel")))
    mean_effort = float(np.mean(values("mean_effort")))
    p95_effort = float(np.mean(values("p95_effort")))
    peak_command = float(np.max(values("peak_command")))
    # Viability gate catches only truly passive submissions (effort < 0.10);
    # the active_control_authority criterion handles the gradual effort quality
    # gradient above the passive floor without double-counting.
    submission_viability_gate = float(
        finite_fraction >= 1.0 and action_fraction >= 1.0 and mean_effort >= 0.10
    )
    mean_jitter = float(np.mean(values("mean_jitter")))
    sat_fraction = float(np.mean(values("sat_fraction")))
    joint_envelope_error = float(
        0.50 * mean_joint + 0.30 * p90_joint + 0.20 * worst_joint
    )
    tip_tail_error = float(0.50 * p90_tip + 0.50 * worst_tip)
    camber_envelope_error = float(max(camber_error, 0.70 * camber_balance))
    joint_shape_score = _lower_better(joint_envelope_error, 0.168, 0.159)
    wing_tip_score = _lower_better(tip_tail_error, 0.172, 0.164)
    recovery_score = _upper_better(fault_recovered, 0.44, 0.52)
    smooth_score = _lower_better(mean_jitter, 0.0058, 0.0046)
    active_authority_score = _upper_better(mean_effort, 0.32, 0.36)
    command_reserve_score = _lower_better(peak_command, 0.993, 0.987)
    root_roll_score = _lower_better(roll_error, 0.130, 0.118)
    camber_alignment_score = _lower_better(camber_envelope_error, 0.073, 0.066)
    saturation_score = _lower_better(sat_fraction, 0.17, 0.149)
    speed_score = _lower_better(max_qvel, 51.3, 50.9)

    # Each envelope score is calibrated from the committed 2026-05-26 oracle
    # proof, with one full/zero band per physical diagnostic. The raw mean,
    # tail, worst-case, and final statistics stay in metadata for auditability,
    # but they are not exposed as separate threshold ladders.
    @rb.criterion(
        id="model_contract",
        weight=0.010,
        description="MJCF compiles with expected morphing-wing joints, actuators, sites, and timestep",
    )
    def _model_contract():
        return model_contract_score

    @rb.criterion(
        id="rollout_validity",
        weight=0.010,
        description="All hidden-case rollouts remain finite with valid length-7 actions",
    )
    def _rollout_validity():
        return rollout_validity_score

    @rb.criterion(
        id="joint_shape_tracking",
        weight=0.080,
        description="Internal spanwise shape residual, excluding root roll and side-average camber, stays within the hidden gust band",
    )
    def _joint_shape_tracking():
        return joint_shape_score

    @rb.criterion(
        id="wing_tip_tracking",
        weight=0.300,
        description="P90 and worst-case wing-tip errors stay within the hidden gust-tail envelope",
    )
    def _wing_tip_tracking():
        return wing_tip_score

    @rb.criterion(id="root_roll_alignment", weight=0.025, description="Root roll follows the commanded morphing attitude")
    def _root_roll_alignment():
        return root_roll_score

    @rb.criterion(id="bilateral_camber_alignment", weight=0.035, description="Left and right camber stay aligned with the commanded wing shape")
    def _bilateral_camber_alignment():
        return camber_alignment_score

    @rb.criterion(
        id="fault_recovery",
        weight=0.080,
        description="Wing-tip error recovers promptly after hidden flap dropouts and gust impulses",
    )
    def _fault_recovery():
        return recovery_score

    @rb.criterion(id="command_smoothness", weight=0.200, description="Command changes stay within the fixed smooth gust-response band")
    def _command_smoothness():
        return smooth_score

    @rb.criterion(id="speed_safety", weight=0.050, description="Joint speeds remain inside the aeroelastic safety envelope")
    def _speed_safety():
        return speed_score

    @rb.criterion(
        id="active_control_authority",
        weight=0.140,
        description="Mean non-passive control authority stays above the gust-rejection floor",
    )
    def _active_control_authority():
        return active_authority_score

    @rb.criterion(
        id="command_reserve",
        weight=0.020,
        description="Peak normalized command retains headroom below the actuator rails",
    )
    def _command_reserve():
        return command_reserve_score

    @rb.criterion(
        id="saturation_reserve",
        weight=0.050,
        description="Active-control saturation fraction remains low across hidden gust and dropout cases",
    )
    def _saturation_reserve():
        return saturation_score

    @rb.penalty(
        id="invalid_or_passive_submission",
        value=-1.0,
        description="Malformed, non-finite, or under-authority passive policies receive no credit",
    )
    def _invalid_or_passive_submission():
        return submission_viability_gate <= 0.0

    rb.metadata["setup_error"] = setup_error
    rb.metadata["case_results"] = [
        {key: value for key, value in row.items() if key not in {"id", "error"}}
        | {"case_index": index}
        for index, row in enumerate(results)
    ]
    rb.metadata["score_interpretation"] = (
        "Ground-truth validation runs solution/solve.sh and is required to "
        "score 1.0. Agent harness submissions use the same deterministic "
        "rubric and should remain below the task difficulty threshold. In "
        "Template Full QA artifacts, ground_truth_result is the oracle proof; "
        "harness_result is a separate non-oracle agent attempt."
    )
    rb.metadata["rubric_design_notes"] = (
        "Joint-shape and wing-tip tracking are distinct: joint shape excludes "
        "root roll and side-average camber before scoring the internal "
        "spanwise residual, while wing-tip tracking validates the stressed "
        "external aerodynamic tail under hidden gust/dropout events. Raw mean "
        "and final tip metrics stay in metadata for auditability, but the "
        "scored tip row emphasizes P90 and worst-case tip error because the "
        "task is a gust-rejection benchmark. Actuator behavior is exposed as "
        "three separate diagnostics: active authority verifies the submission "
        "is not coasting, command reserve checks peak rail headroom, and "
        "saturation reserve checks sustained rail-hitting under hidden faults. "
        "The viability penalty intentionally supersedes the active-authority "
        "row below the documented 0.18 effort floor: below that floor a policy "
        "is treated as coasting rather than actively rejecting hidden gusts."
    )
    rb.metadata["calibration_bands"] = {
        "joint_shape_tracking": {
            "metric": "joint_envelope_error",
            "oracle": ORACLE_CALIBRATION_SUMMARY["joint_envelope_error"],
            "full_credit_at_or_below": 0.159,
            "zero_credit_at_or_above": 0.168,
        },
        "wing_tip_tracking": {
            "metric": "tip_tail_error",
            "oracle": ORACLE_CALIBRATION_SUMMARY["tip_tail_error"],
            "full_credit_at_or_below": 0.164,
            "zero_credit_at_or_above": 0.172,
        },
        "bilateral_camber_alignment": {
            "metric": "camber_envelope_error",
            "oracle": ORACLE_CALIBRATION_SUMMARY["camber_envelope_error"],
            "full_credit_at_or_below": 0.066,
            "zero_credit_at_or_above": 0.073,
        },
        "fault_recovery": {
            "metric": "fault_recovered_fraction",
            "oracle": ORACLE_CALIBRATION_SUMMARY["fault_recovered_fraction"],
            "full_credit_at_or_above": 0.52,
            "zero_credit_at_or_below": 0.44,
        },
        "command_smoothness": {
            "metric": "mean_jitter",
            "oracle": ORACLE_CALIBRATION_SUMMARY["mean_jitter"],
            "full_credit_at_or_below": 0.0046,
            "zero_credit_at_or_above": 0.0058,
        },
        "speed_safety": {
            "metric": "max_qvel",
            "oracle": ORACLE_CALIBRATION_SUMMARY["max_qvel"],
            "full_credit_at_or_below": 50.9,
            "zero_credit_at_or_above": 51.3,
        },
        "active_control_authority": {
            "metric": "mean_effort",
            "oracle": ORACLE_CALIBRATION_SUMMARY["mean_effort"],
            "full_credit_at_or_above": 0.36,
            "zero_credit_at_or_below": 0.32,
        },
        "command_reserve": {
            "metric": "peak_command",
            "oracle": ORACLE_CALIBRATION_SUMMARY["peak_command"],
            "full_credit_at_or_below": 0.987,
            "zero_credit_at_or_above": 0.993,
        },
        "saturation_reserve": {
            "metric": "sat_fraction",
            "oracle": ORACLE_CALIBRATION_SUMMARY["sat_fraction"],
            "full_credit_at_or_below": 0.149,
            "zero_credit_at_or_above": 0.17,
        },
    }
    rb.metadata["recovery_event_definition"] = {
        "tip_error_threshold": RECOVERY_TIP_ERROR_THRESHOLD,
        "coverage_time_seconds": RECOVERY_COVERAGE_TIME,
        "description": (
            "Each dropout or gust event is considered recovered at the first "
            "post-event sample where average wing-tip error is at or below "
            "tip_error_threshold. fault_recovered is the fraction of event "
            "recoveries that occur within coverage_time_seconds."
        ),
    }
    rb.metadata["oracle_calibration_summary"] = ORACLE_CALIBRATION_SUMMARY
    rb.metadata["aggregate_metrics"] = {
        "mean_joint_error": mean_joint,
        "p90_joint_error": p90_joint,
        "worst_case_p90_joint_error": worst_joint,
        "joint_envelope_error": joint_envelope_error,
        "mean_tip_error": mean_tip,
        "p90_tip_error": p90_tip,
        "worst_case_p90_tip_error": worst_tip,
        "tip_tail_error": tip_tail_error,
        "tip_envelope_error": tip_tail_error,
        "roll_error": roll_error,
        "camber_error": camber_error,
        "camber_balance_error": camber_balance,
        "camber_envelope_error": camber_envelope_error,
        "final_tip_error": final_tip,
        "recovery_time": recovery,
        "worst_recovery_time": worst_recovery,
        "fault_recovered": fault_recovered,
        "max_qvel": max_qvel,
        "mean_effort": mean_effort,
        "p95_effort": p95_effort,
        "peak_command": peak_command,
        "submission_viability_gate": submission_viability_gate,
        "model_contract_score": model_contract_score,
        "rollout_validity_score": rollout_validity_score,
        "mean_jitter": mean_jitter,
        "sat_fraction": sat_fraction,
        "joint_shape_score": joint_shape_score,
        "wing_tip_score": wing_tip_score,
        "recovery_score": recovery_score,
        "smooth_score": smooth_score,
        "active_authority_score": active_authority_score,
        "command_reserve_score": command_reserve_score,
        "root_roll_score": root_roll_score,
        "camber_alignment_score": camber_alignment_score,
        "saturation_score": saturation_score,
        "speed_score": speed_score,
    }
    return rb.grade().to_dict()
