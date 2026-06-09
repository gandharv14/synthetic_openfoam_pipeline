# Nonholonomic Trailer Docking

Create a deterministic Python policy at `/tmp/output/policy.py`.

Your policy controls a planar tractor pulling a passive trailer. The tractor has two normalized commands:

```python
def act(obs: dict) -> list[float]:
    return [drive_command, steering_command]
```

Both action values are clipped to `[-1, 1]`. The helper maps them to tractor drive speed and yaw rate for a deterministic MuJoCo-backed tractor-trailer model.

The grader evaluates hidden docking scenarios. In each scenario, the trailer must reach and hold a target pose while the tractor avoids unsafe jackknife angles, stays inside the workspace, and avoids circular no-go regions. Some scenarios require controlled reversing; simply driving the tractor hitch to the target is not enough.

You may use the public files in `data/`, especially `data/trailer_env.py` and `data/public_scenarios.json`, to inspect the observation schema and test your policy. The public helper is available to submitted policies as `trailer_env` during grading. Write final artifacts only under `/tmp/output`.

Important observation fields include:

- `hitch_x`, `hitch_y`, `tractor_yaw`
- `trailer_x`, `trailer_y`, `trailer_yaw`
- `hitch_angle`, `hitch_angle_rate`
- `target_x`, `target_y`, `target_yaw`
- `target_dx`, `target_dy`, `target_yaw_error`
- `desired_hitch_x`, `desired_hitch_y`
- `reverse_required`
- `workspace`, `no_go`, `trailer_length`, `max_drive_speed`, `max_yaw_rate`

The scorer is deterministic. It rewards trailer final pose accuracy, stable final hold, controlled hitch angle, obstacle/workspace safety, useful reverse behavior when requested, smooth actions, and worst-case hidden-scenario robustness. Docking-achievement rows are safety-gated: unsafe jackknife, workspace violations, or no-go violations reduce credit for pose/progress/reverse behavior, and yaw/hold credit also requires meaningful progress toward the target. Scores at or below `0.40` are not normalized upward; the deterministic oracle is calibrated to score `1.0`.

Key public rubric thresholds:

- trailer-center final-window position: full credit at `0.07 m`, zero at `0.75 m`;
- trailer final-window yaw: full credit at `0.10 rad`, zero at `1.10 rad`;
- progress: full credit after closing at least `78%` of initial trailer-target distance, zero below `5%`;
- final hold: measured over the final `0.80 s` using trailer-center speed, with full credit below `0.04 m/s` and zero at `0.42 m/s`; this credit is progress- and safety-gated so standing still away from the dock does not count;
- hitch safety: final hitch angle full credit at `0.10 rad` and zero at `0.55 rad`; post-recovery max hitch full credit at `0.25 rad` and zero at `0.58 rad`; unsafe hitch fraction zeroes out by `10%`;
- reverse behavior: in reverse-required scenarios, combines sustained reverse-action fraction (`82%` full, `52%` zero), target-distance reduction while reversing (`50%` full, `16%` zero), and safe reverse samples;
- worst-case hidden scenario performance is a moderate robustness check to reject policies that solve only easy layouts.
