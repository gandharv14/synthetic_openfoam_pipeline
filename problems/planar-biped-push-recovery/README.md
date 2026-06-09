# Planar Biped Push-Recovery

A MuJoCo policy-control task. The agent writes `/tmp/output/policy.py`; the
grader runs the policy against a fixed planar biped under four deterministic
rollouts plus a feedback-sensitivity probe.

The morphology is fixed (`data/planar_biped.xml`) so the task isolates
*control* skill from morphology hacks: there is no flat-puck workaround, no
weld-to-world trick, no projectile geometry, no hidden support — the agent
only chooses how to map observations to position-actuator targets.

## Layout

```
problems/planar-biped-push-recovery/
├── README.md, instruction.md, metadata.json, task.toml
├── data/planar_biped.xml              # fixed biped model (also mounted at /data/ in the runtime)
├── environment/Dockerfile             # runtime image (uses the shared MuJoCo base)
├── scorer/
│   ├── compute_score.py               # deterministic grader (14 criteria)
│   └── data/eval_cases.json           # hidden initial states / pushes / friction
├── solution/
│   ├── solve.sh                       # writes the oracle policy.py
│   ├── render.sh                      # renders the rubric reviewer video
│   └── render_config.py               # initial state, push, and camera for rendering
├── baselines/naive.sh                 # zero-action baseline (scores ≤ 0.25)
└── tests/test.sh                      # local smoke tests
```

## Rubric

14 deterministic criteria, weighted, summing to 1.0 after normalization:

| Group        | Criterion                  | What it measures                                       |
| ------------ | -------------------------- | ------------------------------------------------------ |
| Structural   | `policy_file_exists`       | `/tmp/output/policy.py` is present                     |
| Structural   | `policy_action_valid`      | `act(obs)` returns a finite length-6 action            |
| Feedback     | `feedback_sensitive`       | hip/ankle command varies with torso pitch (\|Δ\| > 0.05) |
| Feedback     | `stabilizing_feedback_sign`| Δ has the sign that restores upright pose              |
| Anchor       | `fixed_model_sanity`       | hidden model has nq=9, nv=9, nu=6                      |
| Quiet stand  | `nominal_standing_height`  | torso z ≥ 1.03 m throughout                            |
| Quiet stand  | `nominal_pose_bound`       | \|pitch\| ≤ 0.02, \|drift\| ≤ 0.02 throughout              |
| Quiet stand  | `nominal_quietness`        | peak \|qvel\| ≤ 3.0 rad/s (no jitter)                    |
| Push +x      | `left_push_survival`       | z ≥ 1.02 m and peak \|pitch\| ≤ 0.18 rad during push     |
| Push +x      | `left_push_recovery`       | final \|pitch\| ≤ 0.04 rad and final \|drift\| ≤ 0.10 m    |
| Push −x      | `right_push_survival`      | mirror of left survival                                |
| Push −x      | `right_push_recovery`      | mirror of left recovery                                |
| Friction     | `low_friction_robustness`  | 0.72×μ + 30 N push: \|pitch\| ≤ 0.07, \|drift\| ≤ 0.10     |
| Global       | `all_rollouts_finite`      | no NaN, peak \|qvel\| ≤ 8.0 across every rollout         |

## Hidden evaluation cases

Defined in `scorer/data/eval_cases.json`:

| Case                    | Duration | Initial pitch | Push                  | Friction |
| ----------------------- | -------- | ------------- | --------------------- | -------- |
| `nominal_stand`         | 3.0 s    | 0.000         | none                  | 1.00×    |
| `left_push_recovery`    | 3.5 s    | +0.040        | +42 N, t=0.70, 0.12 s | 1.00×    |
| `right_push_recovery`   | 3.5 s    | −0.040        | −42 N, t=0.70, 0.12 s | 1.00×    |
| `low_friction_small_push` | 3.5 s  | +0.025        | +30 N, t=0.90, 0.10 s | 0.72×    |

All cases use the same pinned physics settings declared in the XML
(`timestep=0.002`, `integrator=implicitfast`, `iterations=80`,
`tolerance=1e-9`) and reset state from a fixed `initial_qpos`, so scores
are reproducible bit-for-bit across runs.

## Oracle

`solution/solve.sh` emits a hand-tuned PD feedback policy:

```
hip = ankle = 0.08 + 1.8·pitch + 0.3·pitch_rate + 0.1·x_rate − 0.3·x
knee = −0.18  (constant bent stance)
```

Gains were found by grid search over `(offset, kp_pitch, kd_pitch, k_xv,
k_x, knee_target)` and selected for symmetric robustness (both sides),
tight quiet-stand bounds, and safety margin above the rubric's 42 N
push — the same policy still recovers symmetrically at 50 N pushes
(verified) and only fails when one side exceeds 55 N.

## Expected scores

- **Oracle** (`solution/solve.sh` → grader): **1.00**
- **Naive zero-action** (`baselines/naive.sh` → grader): **0.21**
- **Default no-policy / noop runtime**: **0.03**

The 0.21 floor for the naive zero-action policy comes from the fact that
the bent-stance is metastable at zero ctrl: the biped happens to stay upright
in quiet stand but oscillates noticeably, fails the tightened pose-bound and
quietness criteria, and collapses under every push.

## Local validation

From the repository root:

```bash
# Smoke test the oracle against the scorer (no Docker):
problems/planar-biped-push-recovery/solution/solve.sh
PYTHONPATH=grader/src uv run python -c "
from pathlib import Path
import sys; sys.path.insert(0, 'problems/planar-biped-push-recovery/scorer')
from compute_score import compute_score
print(compute_score(Path('/tmp/output'), None, Path('problems/planar-biped-push-recovery/scorer/data'))['score'])
"

# Full ground-truth proof (requires Docker):
uv run lbx-rl-harness run --runtime ground-truth --problem-dir problems/planar-biped-push-recovery
```

The reviewer video is produced by `solution/render.sh` and saved as
`/tmp/output/rendering.mp4` (1280×720, 5.0 s). The render scenario mirrors
the `left_push_recovery` grader case (42 N · 0.12 s) so it cannot be
inconsistent with what the rubric grades.
