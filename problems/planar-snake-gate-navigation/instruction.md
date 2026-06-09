# Planar Snake Gate Navigation

Write a deterministic Python policy at `/tmp/output/policy.py`.

Your module must expose one of:

- `act(obs)`
- `get_action(obs)`
- `class Policy` with an `act(obs)` method

Each call receives an observation dictionary and must return an action vector:

```text
[drive, turn, joint0, joint1, joint2, joint3, joint4, joint5]
```

All values are clipped to roughly `[-1, 1]`. `drive` produces forward body
force along the snake head direction, `turn` produces yaw torque at the head,
and each `joint*` value is a target angle for one hinge in the articulated body.

Important observation fields:

- `time`: rollout time in seconds.
- `action_size`: expected action length, usually `8`.
- `num_joints`: number of joint targets, usually `6`.
- `head_xy`, `head_yaw`: snake head pose.
- `head_velocity_world`, `head_velocity_body`: head velocity.
- `tail_xy`: approximate tail endpoint.
- `joint_angles`, `joint_velocities`: current articulated state.
- `body_points`: sampled whole-body points for self-checking clearance.
- `gate_index`, `num_gates`: ordered gate progress.
- `target_gate`: current gate dictionary with `center`, `yaw`, `width`, and `depth`.
- `next_gate`: next gate dictionary, or `null`.
- `final_target`: final target point.
- `no_go`: circular no-go regions visible during the rollout.
- `workspace`: workspace bounds.

The hidden grader uses deterministic MuJoCo rollouts. It varies gate layouts,
friction, damping, no-go regions, and disturbances. Score comes from ordered
gate progress, closest gate approach, final target distance, heading alignment,
whole-body workspace/no-go clearance, coherent joint motion, moderate gait
activity, smooth bounded actions, and worst-case hidden-scenario robustness.

Public helpers and example scenarios are available in `/data`.
