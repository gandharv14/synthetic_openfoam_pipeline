# GPU Morphing Wing Gust Rejection

Train, tune, or author a feedback policy for a redundant articulated morphing
wing. The fixed MuJoCo model is available at:

```text
/data/morphing_wing.xml
```

Write exactly:

```text
/tmp/output/policy.py
```

The policy module must expose either:

```python
def act(obs: dict) -> list[float]:
    ...
```

or:

```python
class Policy:
    def act(self, obs: dict) -> list[float]:
        ...
```

## System

The wing has seven revolute joints:

```text
root_roll,
left_inner, left_mid, left_tip,
right_inner, right_mid, right_tip
```

The action is a length-7 sequence of normalized motor commands in `[-1, 1]`,
in the same order. The grader clips actions before applying hidden actuator
effects, but non-finite or badly shaped actions lose score.

## Observation

Each call receives a public observation dictionary:

```python
{
    "time": float,
    "step": int,
    "qpos": np.ndarray,
    "qvel": np.ndarray,
    "left_tip_pos": np.ndarray,
    "right_tip_pos": np.ndarray,
    "target_left_tip_pos": np.ndarray,
    "target_right_tip_pos": np.ndarray,
    "target_root_roll": float,
    "target_left_camber": float,
    "target_right_camber": float,
    "last_ctrl": np.ndarray,
    "phase": float,
}
```

Hidden evaluation cases vary hinge stiffness, joint damping, asymmetric payload
mass, actuator gains, brief flap dropouts, and gust-like joint torques. The
exact schedules are hidden. Your policy must adapt from the live observation
stream rather than replaying a fixed action sequence.

The wing is an aeroelastic shape-control system, so average joint tracking is
not sufficient on its own. High-value tracking requires smooth command updates,
low P90 and worst-case tip error, active control authority, command reserve,
low actuator saturation, and robust completion across the hidden gust/dropout cases.
The grader uses one
calibrated full/zero band for each physical diagnostic rather than many
independent threshold rows: internal joint-shape tracking is scored as a
consolidated mean/tail/worst spanwise residual after removing root roll and
side-average camber error, wing-tip tracking is scored as a stressed P90/worst
tail envelope, and camber tracking is scored by
the larger of commanded-camber residual and scaled left/right imbalance.
Smoothness and speed safety remain separate diagnostics, and the rubric also
checks the fixed MJCF/site/action contract plus finite hidden-case rollouts as
small structural rows. Actuator use is scored as three separate diagnostics:
mean active authority, peak-command reserve, and low active saturation. This
keeps a policy that coasts, rail-hits, or exhausts command headroom from hiding
behind a single combined actuator row. For recovery scoring, each hidden
dropout or gust event starts a
timer, and the event is counted as recovered only if average wing-tip error
returns below the fixed recovery tolerance within the fixed coverage window.
The committed oracle proof provides public calibration context: the oracle's
aggregate envelope values are approximately `0.151` for joint shape, `0.156`
for the P90/worst wing-tip tail, `0.063` for camber, `0.0044` mean command
jitter, `50.7` maximum joint-speed norm, `0.388` mean effort, `0.985` peak
command, and `0.142` active-control saturation fraction. The deterministic
full/zero bands are recorded in scorer metadata; the smaller rows include
explicit headroom beyond those oracle metrics, for example camber is full at
or below `0.068`, mean jitter at or below `0.0052`, max speed at or below
`51.0`, and saturation at or below `0.18`.

## GPU Requirement

This is a policy-training task. The intended workflow is to train or tune a
neural/residual controller with batched randomized rollouts on the requested
GPU, then export deterministic inference code to `/tmp/output/policy.py`.
Public examples in `/data/public_training_cases.json` show the case format, and
`/data/policy_template.py` gives a minimal callable policy shell. The private
grader uses separate hidden cases with different payload, stiffness, dropout,
and gust schedules.

## Scoring

The hidden grader runs deterministic MuJoCo rollouts with fixed seeds and
scores separate criteria for:

- policy interface and action validity,
- fixed MJCF/site/actuator contract,
- finite rollout state under all hidden cases,
- consolidated joint-shape envelope tracking,
- stressed P90/worst wing-tip tail tracking,
- root-roll attitude tracking with smooth aeroelastic control,
- camber and symmetry tracking with smooth aeroelastic control,
- recovery after dropouts and impulse disturbances,
- bounded joint speeds,
- active control authority, peak command reserve, saturation reserve, and
  smooth command jitter,
- robustness across stiffness, fatigue, payload, dropout, and gust scenarios.

Only files under `/tmp/output` are graded.
