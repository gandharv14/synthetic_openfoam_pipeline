# Nonholonomic Trailer Docking

Write `/tmp/output/policy.py` for a planar tractor pulling a passive trailer. The policy must dock the trailer, not just the tractor hitch, across hidden MuJoCo-backed deterministic scenarios with reversing, angled bays, no-go regions, jackknife starts, and final-hold disturbances.

The public helper in `data/trailer_env.py` exposes the observation/action schema and deterministic kinematic trailer stepping used by the scorer. It is importable as `trailer_env` from submitted policies during grading. The MuJoCo model provides the robot geometry, named bodies/sites, and reviewer rendering; the rollout state is advanced by the helper so contact/wheel constraint noise does not dominate the task.

The grader loads hidden scenarios from `scorer/data/hidden_scenarios.json`, imports the submitted policy through `PolicyWorker`, runs fixed rollouts, and scores final trailer pose, docking hold stability, hitch/jackknife safety, no-go/workspace clearance, useful reverse-control behavior, smoothness, and worst-case hidden performance.

The public weighted rubric is aligned with the headline score: component criteria carry most of the score, low-articulation hitch safety and sustained useful reversing are emphasized because they distinguish stable trailer docking from hitch-only motion or short reverse bursts, and `worst_case` is a robustness check for policies that fail one hidden family. Hitch safety and reverse behavior expose their key thresholds in `instruction.md` while hidden scenario layouts remain private. Scores at or below `0.40` are left unchanged; the deterministic oracle raw headline is normalized to `1.0` to satisfy the MuJoCo ground-truth contract.

Local iteration targets:

- oracle/reference should score `1.0`;
- missing policy should score `0.0`;
- noop and direct hitch/target baselines should remain below `0.40`;
- official PR readiness still requires ground-truth harness/build proof and provider-backed agent scoring.
