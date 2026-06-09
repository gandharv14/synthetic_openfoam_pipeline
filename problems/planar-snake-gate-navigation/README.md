# Planar Snake Gate Navigation

This task asks agents to write `/tmp/output/policy.py` for an articulated
MuJoCo snake. The policy controls forward drive, yaw torque, and six joint
targets while the hidden scorer evaluates deterministic gate sequences with
shifted friction, no-go regions, narrow openings, and disturbances.

The scorer rewards ordered gate completion, gate accuracy, final target hold,
heading alignment, whole-body clearance, coherent articulated motion, moderate
gait activity, smooth controls, and worst-case robustness.

## Calibration

`solution/solve.sh` is the oracle submission. Its hidden-scenario raw headline
is `0.6599347862038533`, which the scorer calibrates to a reported ground-truth
score of `1.0`.

Agent harness scores are separate difficulty evidence and are expected to stay
below the `0.40` acceptance cutoff. Recent checks scored OpenClaw at `0.2018`
locally and Claude deepagents at `0.264` in GitHub Full QA; those are not oracle
scores.

Run focused local checks from the repository root:

```bash
uv run lbx-rl-harness run --problem-dir problems/planar-snake-gate-navigation --runtime ground-truth
```
