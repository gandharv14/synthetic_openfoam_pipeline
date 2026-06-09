#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
MODEL_SRC="${MORPHING_WING_XML:-/data/morphing_wing.xml}"
if [[ ! -f "${MODEL_SRC}" && -n "${BASH_SOURCE[0]:-}" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  MODEL_SRC="${SCRIPT_DIR}/../data/morphing_wing.xml"
fi
if [[ ! -f "${MODEL_SRC}" && -f "data/morphing_wing.xml" ]]; then
  MODEL_SRC="data/morphing_wing.xml"
fi
if [[ ! -f "${MODEL_SRC}" && -f "problems/gpu-morphing-wing-gust-rejection/data/morphing_wing.xml" ]]; then
  MODEL_SRC="problems/gpu-morphing-wing-gust-rejection/data/morphing_wing.xml"
fi
if [[ ! -f "${MODEL_SRC}" ]]; then
  echo "morphing_wing.xml not found for oracle packaging" >&2
  exit 1
fi
OUTPUT_DATA_DIR="${OUTPUT_DIR}/data"
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DATA_DIR}"
cp "${MODEL_SRC}" "${OUTPUT_DATA_DIR}/morphing_wing.xml"

cat > "${OUTPUT_DIR}/policy.py" <<'PY'
"""Closed-loop oracle for the morphing-wing gust rejection task.

Every command is computed from the current public observation. The controller
uses MuJoCo inverse dynamics on the public model plus PID feedback and live
wing-tip residuals; it does not know hidden gust, dropout, payload, or fatigue
schedules.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np


class Policy:
    KP = np.array([112.0, 96.0, 82.0, 64.0, 96.0, 82.0, 64.0])
    KD = np.array([23.0, 18.0, 15.0, 10.5, 18.0, 15.0, 10.5])
    KI = np.array([30.0, 25.0, 20.0, 13.0, 25.0, 20.0, 13.0])
    INTEG_LIMIT = np.array([0.20, 0.22, 0.22, 0.18, 0.22, 0.22, 0.18])
    ALPHA = 0.76

    def __init__(self):
        candidates = [
            Path(os.environ["MORPHING_WING_XML"]) if "MORPHING_WING_XML" in os.environ else None,
            Path("/data/morphing_wing.xml"),
            Path(__file__).resolve().parent / "data" / "morphing_wing.xml",
            Path(__file__).resolve().parent.parent / "data" / "morphing_wing.xml",
            Path("data/morphing_wing.xml"),
            Path.cwd() / "data" / "morphing_wing.xml",
            Path.cwd() / "problems" / "gpu-morphing-wing-gust-rejection" / "data" / "morphing_wing.xml",
        ]
        model_path = next((p for p in candidates if p is not None and p.exists()), None)
        if model_path is None:
            raise FileNotFoundError("morphing_wing.xml not found")
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.gear = np.array([float(self.model.actuator_gear[i, 0]) for i in range(self.model.nu)])
        self.left_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_tip_site")
        self.right_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_tip_site")
        self.qmin = self.model.jnt_range[:, 0].copy()
        self.qmax = self.model.jnt_range[:, 1].copy()
        self.integral = np.zeros(self.model.nv)
        self.last_time = -1.0
        self.last_ctrl = np.zeros(self.model.nu)
        self.last_ref = None

    def _inverse(self, q, qd, qdd):
        data = mujoco.MjData(self.model)
        data.qpos[:] = q
        data.qvel[:] = qd
        data.qacc[:] = qdd
        mujoco.mj_inverse(self.model, data)
        return data.qfrc_inverse.copy()

    def _solve_reference(self, q, target_left, target_right, target_root, target_left_camber, target_right_camber):
        q_ref = np.clip(q.copy(), self.qmin, self.qmax)
        root_row = np.zeros((1, self.model.nv))
        root_row[0, 0] = 1.0
        left_camber_row = np.zeros((1, self.model.nv))
        left_camber_row[0, 1:4] = 1.0 / 3.0
        right_camber_row = np.zeros((1, self.model.nv))
        right_camber_row[0, 4:7] = 1.0 / 3.0

        for _ in range(10):
            data = mujoco.MjData(self.model)
            data.qpos[:] = q_ref
            data.qvel[:] = 0.0
            mujoco.mj_forward(self.model, data)

            left = data.site_xpos[self.left_site].copy()
            right = data.site_xpos[self.right_site].copy()
            jac_left = np.zeros((3, self.model.nv))
            jac_right = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, data, jac_left, jac_rot, self.left_site)
            mujoco.mj_jacSite(self.model, data, jac_right, jac_rot, self.right_site)

            residual = np.concatenate(
                [
                    target_left - left,
                    target_right - right,
                    [target_root - q_ref[0]],
                    [target_left_camber - float(np.mean(q_ref[1:4]))],
                    [target_right_camber - float(np.mean(q_ref[4:7]))],
                ]
            )
            jac = np.vstack([jac_left, jac_right, root_row, left_camber_row, right_camber_row])
            weights = np.array([1.0, 1.0, 1.35, 1.0, 1.0, 1.35, 0.95, 0.58, 0.58])
            jw = jac * weights[:, None]
            rw = residual * weights
            gram = jw @ jw.T + 2.5e-3 * np.eye(jw.shape[0])
            dq = jw.T @ np.linalg.solve(gram, rw)
            q_ref = np.clip(q_ref + np.clip(dq, -0.18, 0.18), self.qmin, self.qmax)
            if np.linalg.norm(residual) < 0.015:
                break
        return q_ref

    def act(self, obs):
        q = np.asarray(obs["qpos"], dtype=float)
        qd = np.asarray(obs["qvel"], dtype=float)
        target_left = np.asarray(obs["target_left_tip_pos"], dtype=float)
        target_right = np.asarray(obs["target_right_tip_pos"], dtype=float)
        target_root = float(obs["target_root_roll"])
        target_left_camber = float(obs["target_left_camber"])
        target_right_camber = float(obs["target_right_camber"])
        t = float(obs["time"])

        if t <= 1e-9 or t < self.last_time:
            self.integral[:] = 0.0
            self.last_ctrl[:] = 0.0
            self.last_ref = None
        dt = 0.004 if self.last_time < 0.0 else max(1e-4, min(0.02, t - self.last_time))
        self.last_time = t

        target = self._solve_reference(q, target_left, target_right, target_root, target_left_camber, target_right_camber)
        if self.last_ref is None:
            target_vel = np.zeros_like(target)
        else:
            target_vel = np.clip((target - self.last_ref) / dt, -4.0, 4.0)
        self.last_ref = target.copy()

        err = target - q
        derr = target_vel - qd
        if np.linalg.norm(err) < 0.55:
            self.integral += err * dt
            self.integral = np.clip(self.integral, -self.INTEG_LIMIT, self.INTEG_LIMIT)
        else:
            self.integral *= 0.85

        qdd_des = self.KP * err + self.KD * derr + self.KI * self.integral
        qdd_des = np.clip(qdd_des, -100.0, 100.0)
        tau = self._inverse(q, qd, qdd_des)
        ctrl = tau / self.gear

        left = np.asarray(obs.get("left_tip_pos", [0.0, 0.0, 0.0]), dtype=float)
        right = np.asarray(obs.get("right_tip_pos", [0.0, 0.0, 0.0]), dtype=float)
        target_left = np.asarray(obs.get("target_left_tip_pos", left), dtype=float)
        target_right = np.asarray(obs.get("target_right_tip_pos", right), dtype=float)
        left_z = float(target_left[2] - left[2])
        right_z = float(target_right[2] - right[2])
        left_y = float(target_left[1] - left[1])
        right_y = float(target_right[1] - right[1])

        ctrl[0] += 0.06 * (left_z - right_z)
        ctrl[1:4] += np.array([0.10, 0.075, 0.050]) * left_z + np.array([0.030, 0.020, 0.015]) * left_y
        ctrl[4:7] += np.array([0.10, 0.075, 0.050]) * right_z - np.array([0.030, 0.020, 0.015]) * right_y

        ctrl = np.clip(ctrl, -0.985, 0.985)
        smooth = self.ALPHA * ctrl + (1.0 - self.ALPHA) * self.last_ctrl
        smooth = np.clip(smooth, -0.985, 0.985)
        self.last_ctrl = smooth.copy()
        return smooth.tolist()


_POLICY = Policy()


def act(obs):
    return _POLICY.act(obs)
PY

cat > "${OUTPUT_DIR}/README.md" <<'MD'
Oracle policy: MuJoCo inverse-dynamics PID with live wing-tip residual feedback.
The controller is closed-loop and uses only public observation keys.
MD

echo "Wrote oracle policy to ${OUTPUT_DIR}/policy.py"
