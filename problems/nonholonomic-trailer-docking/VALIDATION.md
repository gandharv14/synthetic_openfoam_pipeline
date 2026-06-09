# Nonholonomic Trailer Docking Validation

Status: submission-ready locally after reviewer-risk and difficulty hardening.
The oracle scores exactly `1.0`, weak/adversarial baselines remain below
`0.40`, and a fresh OpenClaw-backed local proxy harness attempt scored
`0.287253`.

This note is local handoff evidence. Official acceptance still depends on the
normal PR/mothership/customer validation path.

## Current Local Validation

Cheap local checks passed on 2026-05-18:

```bash
uv run python -m py_compile \
  problems/nonholonomic-trailer-docking/data/trailer_env.py \
  problems/nonholonomic-trailer-docking/scorer/compute_score.py \
  problems/nonholonomic-trailer-docking/solution/render_config.py

uv run python - <<'PY'
import json, tomllib
from pathlib import Path
base = Path("problems/nonholonomic-trailer-docking")
tomllib.loads((base / "task.toml").read_text())
for p in [base / "metadata.json", base / "data/public_scenarios.json", base / "scorer/data/hidden_scenarios.json"]:
    json.loads(p.read_text())
print("static_parse_ok")
PY

bash -n problems/nonholonomic-trailer-docking/solution/solve.sh \
  problems/nonholonomic-trailer-docking/solution/render.sh \
  problems/nonholonomic-trailer-docking/baselines/noop.sh \
  problems/nonholonomic-trailer-docking/baselines/direct_hitch.sh \
  problems/nonholonomic-trailer-docking/baselines/bang_bang.sh \
  problems/nonholonomic-trailer-docking/baselines/public_replay.sh \
  problems/nonholonomic-trailer-docking/baselines/final_pose_only.sh \
  problems/nonholonomic-trailer-docking/tests/test.sh
```

Direct scorer sweep after reviewer-risk hardening, no Docker/provider/API:

| Submission | Score | Raw headline | Avg scenario | Worst scenario |
| --- | ---: | ---: | ---: | ---: |
| oracle/reference | `1.000000` | `0.856390` | `0.829884` | `0.701129` |
| noop | `0.287253` | `0.287253` | `0.307090` | `0.018779` |
| direct_hitch | `0.105351` | `0.105351` | `0.089397` | `0.051578` |
| final_pose_only | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| bang_bang | `0.059833` | `0.059833` | `0.001670` | `0.000000` |
| public_replay | `0.000000` | `0.000000` | `0.000000` | `0.000000` |
| saved OpenClaw high policy before hardening | `0.285966` | `0.285966` | `0.359811` | `0.060025` |
| latest OpenClaw proxy agent policy | `0.287253` | `0.287253` | `0.307090` | `0.018779` |
| missing_policy | `0.000000` | n/a | n/a | n/a |

Interpretation: the strongest observed non-oracle policies are below the target
`0.40` cutoff after the low-articulation/reverse-control hardening. The oracle
raw headline is normalized to `1.0`; scores at or below `0.40` are unchanged by
calibration.

## Scorer Hardening Notes

The latest scorer pass directly addressed the remaining rubric-quality risks:

- removed `task_completion` as a weighted public rubric row;
- changed `hold` to use trailer-center speed but require real progress and
  safety validity, so standing still away from the dock does not earn hold
  credit;
- changed `reverse_control` from reverse-action fraction only to a useful
  reverse metric combining reverse fraction, target-distance reduction during
  reverse samples, and safe reverse samples;
- safety-gated the docking achievement rows so pose/progress/reverse credit is
  reduced when a policy jackknifes, leaves the workspace, or enters no-go
  regions;
- progress-gated orientation and hold credit so a policy cannot earn yaw/hold
  credit by remaining near its initial pose;
- rebalanced weights and tightened low-articulation/reverse/smoothness
  thresholds after a local proxy agent found a `0.794071` policy that solved
  several scenarios with short reverse windows, larger hitch articulation, and
  high control slew;
- exposed high-weight hitch and reverse thresholds in the public instruction;
- kept `policy_present` zero-weight for valid submissions;
- kept non-finite values scoring `0.0`;
- kept metadata explaining that scores at or below `0.40` are not normalized
  upward.

Current weighted rubric rows:

| Criterion | Weight |
| --- | ---: |
| position | `0.130` |
| orientation | `0.065` |
| progress | `0.130` |
| hold | `0.045` |
| hitch_safety | `0.210` |
| workspace | `0.025` |
| no_go | `0.025` |
| reverse_control | `0.190` |
| smoothness | `0.060` |
| worst_case | `0.120` |
| policy_present | `0.000` |

## Harness And Proof

After reviewer-video and scorer hardening, the OpenClaw-backed local harness
proxy completed below `0.40`:

```bash
AGENT_TIMEOUT_SEC=900 .harness-runs/run_local_harness_openclaw.sh problems/nonholonomic-trailer-docking
```

Result:

| Evidence | Result |
| --- | --- |
| Ground truth | `1.000000` |
| OpenClaw/deepagents local proxy harness | `0.28725346153846154` |
| Rubric quality | `completed`; remaining warnings are intentional safety/progress gating, some overlap in `worst_case`, and prescriptive but public reverse/smoothness thresholds |
| `verify_build_proof` | `ok=True`, no errors |
| `TaskValidator` | `valid` |

Post-run proof check:

```bash
uv run python - <<'PY'
from pathlib import Path
from alignerr_plugin.proof import verify_build_proof
ok, errors, proof = verify_build_proof(Path("problems/nonholonomic-trailer-docking"))
print(ok, errors)
print((proof or {}).get("ground_truth_result", {}).get("score"))
print(((proof or {}).get("harness_result") or {}).get("score"))
PY
```

Reviewer render command can also be smoke-tested without Docker/provider/API:

```bash
tmpdir=$(mktemp -d)
LBT_OUTPUT_DIR="$tmpdir" bash solution/render.sh
ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$tmpdir/rendering.mp4"
```

Observed render result after the reviewer-video correction: non-empty MP4,
`1280,720`, `5.8 s`.

The reviewer video uses a straight reverse docking scenario chosen so the
oracle visibly reaches the green target marker. Direct rollout diagnostics for
that scenario:

| Metric | Value |
| --- | ---: |
| final trailer-center error | `0.037719 m` |
| final yaw error | `0.072479 rad` |
| final hitch angle | `0.032292 rad` |
| minimum no-go clearance | `0.123886 m` |
| minimum workspace margin | `0.092717 m` |

Reviewer render clarity:

- red translucent disks show no-go regions;
- green translucent trailer footprint shows the target trailer center pose;
- green center marker and yaw axis show the desired target orientation.

## PR Blockers

- Official/default provider-backed harness or mothership validation remains the
  authoritative difficulty signal. Local OpenClaw proxy evidence is useful but
  not identical to the official provider route.
- Include the regenerated `.alignerr/build_proof.json` and
  `.alignerr/ground_truth/rendering.mp4` if this task is submitted.
