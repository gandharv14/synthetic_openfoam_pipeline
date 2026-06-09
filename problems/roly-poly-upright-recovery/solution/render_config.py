from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from lbx_rl_tasks_harness.render_mujoco import apply_action

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
if str(DATA_DIR) not in sys.path:
    sys.path.insert(0, str(DATA_DIR))

from roly_env import apply_scenario, observation as roly_observation, reset_state  # noqa: E402

RENDER_SCENARIO = json.loads(
    (Path(__file__).resolve().parents[1] / "scorer/data/hidden_scenarios.json").read_text()
)[0]


def initialize(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    model.vis.global_.offwidth = 1280
    model.vis.global_.offheight = 720
    apply_scenario(model, RENDER_SCENARIO)
    reset_state(model, data, RENDER_SCENARIO)


def before_step(model, data, policy) -> None:
    if policy is None:
        return
    obs = roly_observation(model, data, RENDER_SCENARIO, float(data.time))
    try:
        action = policy.act(obs)
    except Exception:
        action = policy(obs)
    apply_action(model, data, action)
