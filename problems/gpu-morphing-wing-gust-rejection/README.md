# GPU Morphing Wing Gust Rejection

This MuJoCo task asks agents to export a policy for a fixed seven-joint
articulated morphing wing. The policy tracks commanded roll, dihedral, and
camber trajectories while hidden grading cases vary hinge stiffness, damping,
actuator effectiveness, flap dropouts, payload balance, and gust torques.

The task requests one H100 because the intended solver workflow is GPU-backed
policy training or residual-policy tuning over randomized rollout batches.
Ground-truth verification stays deterministic and fast by exporting a closed-loop
oracle policy through `solution/solve.sh`.

Key acceptance properties:

- `task.toml` declares `[difficulty].task_type = "mujoco"` and GPU resources.
- The grader uses `PolicyWorker`; committed private `scorer/data/hidden_cases.json`
  schedules stay private in `/mcp_server/data` and the scorer fails closed if
  that hidden fixture is missing.
- The oracle computes every command from the current public observation and
  does not replay hardcoded time windows or action schedules.
- The rubric has 12 deterministic criteria across MJCF/site contract checks,
  finite hidden-case rollouts, internal spanwise joint-shape tracking,
  stressed wing-tip tail tracking, roll and camber alignment, gust/dropout recovery,
  active-control smooth commands, speed safety, active control authority,
  peak-command reserve, and saturation reserve. The
  invalid/passive penalty enforces the policy rollout contract. Thresholds are tied to the committed
  `ground_truth_result` metrics with proportional partial-credit ranges around
  the oracle; joint and wing-tip tracking stay separate so reviewers can see
  whether a policy failed the internal shape envelope or the external
  aerodynamic tip envelope. The internal joint-shape envelope removes root-roll
  error and side-average camber residual before scoring the spanwise shape
  residual, leaving those signals to their own roll and camber rows. Wing-tip
  tail tracking carries the largest single weight because hidden gust/dropout
  rejection is the external aerodynamic outcome, but it is capped at 0.30 so
  joint shape, recovery, camber, speed, smoothness, and actuator diagnostics
  retain substantial diagnostic weight. Its scored envelope uses P90 and
  worst-case tip errors; mean and final tip statistics remain in metadata for
  auditability rather than duplicate scoring rows.
  Internal joint-shape
  tracking, wing-tip control, camber alignment, recovery, smoothness, speed
  safety, and actuator-use diagnostics carry the primary score. Recovery is
  weighted as a robustness row for hidden gust/dropout events, and command
  margin is weighted as a real actuator check,
  with each dropout or gust event counted only when average wing-tip error
  returns below the fixed recovery tolerance inside the fixed coverage window.
  Actuator use is separated into mean active authority, peak-command reserve,
  and saturation reserve, so coasting, rail-hitting commands, and exhausted
  headroom are visible as independent deterministic rows.
- The committed `.alignerr/build_proof.json` records the ground-truth oracle
  run from `solution/solve.sh`, including the 1.0 score and 1280x720 reviewer
  video metadata. CI may also publish separate agent `harness_result` artifacts;
  those are non-oracle attempts and are not the ground-truth proof.
- The prompt describes the physical diagnostics qualitatively while the scorer
  metadata records the oracle-calibrated aggregate metrics and envelope scores
  used for review.
- Public oracle calibration context is recorded in both the scorer metadata and
  the prompt: the committed oracle proof has joint envelope about `0.151`, tip
  tail about `0.156`, camber envelope about `0.063`, mean jitter about
  `0.0044`, max joint-speed norm about `50.7`, mean effort about `0.388`, peak
  command about `0.985`, and active-control saturation fraction about `0.142`.
  The scorer metadata also records the full/zero bands for every calibrated
  diagnostic; the smaller rows have explicit headroom beyond the oracle, such
  as camber full credit at `0.068`, mean jitter at `0.0052`, speed at `51.0`,
  and saturation fraction at `0.18`.
- `solution/render.sh` writes `/tmp/output/rendering.mp4` through the shared
  MuJoCo renderer.
