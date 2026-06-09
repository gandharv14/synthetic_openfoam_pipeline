# Planar Biped Push-Recovery Policy

Author a Python feedback policy that keeps a fixed MuJoCo planar biped
upright through deterministic lateral perturbations.

## Output contract

Write your policy to:

```text
/tmp/output/policy.py
```

The module must expose **either**:

```python
def act(obs):
    ...
```

**or**:

```python
class Policy:
    def act(self, obs):
        ...
```

`act` is called every 5 simulation steps (~100 Hz, with the model running at
500 Hz). It must return a sequence of **six finite floats** — target joint
angles, in radians — in this order:

```text
[left_hip, left_knee, left_ankle, right_hip, right_knee, right_ankle]
```

The grader clips each command to the actuator `ctrlrange` declared in the
XML, so values outside the limits are not an error but won't give you extra
authority. Position actuators apply torque proportional to (target − current
joint angle), so think of your output as a *target pose*, not a torque.

## Observation contract

`act` receives a dict shaped like:

```python
{
    "time": float,                # simulation time in seconds
    "step": int,                  # simulation step index
    "qpos": np.ndarray,           # length 9: [root_x, root_z, root_pitch, l_hip, l_knee, l_ankle, r_hip, r_knee, r_ankle]
    "qvel": np.ndarray,           # length 9: matching velocity order
    "sensordata": np.ndarray,     # IMU + joint sensors (see model XML for layout)
    "ctrl": np.ndarray,           # length 6: last applied actuator command
    "nu": 6, "nq": 9, "nv": 9,
}
```

`qpos[2]` is torso pitch in radians (positive = leaning forward in +x).
`qvel[0]` is horizontal CoM velocity. These two are sufficient for a basic
stabilizing PD controller; the other entries let you go further if you want.

## What is graded

The hidden grader runs four deterministic 3–3.5 s rollouts:

1. **Quiet stand** — no push, evaluates pose holding and joint velocity quietness.
2. **Left push** — +42 N torso force for 0.12 s, then 2.7 s of recovery time.
3. **Right push** — −42 N torso force for 0.12 s, then 2.7 s of recovery time.
4. **Low-friction push** — floor friction at 72% of nominal, 30 N push.

In addition, the policy is probed at three off-equilibrium poses
(pitch=0, +0.10, −0.10 rad) to check that it produces sign-correct,
non-trivial feedback (a constant action fails this check by construction).

You will be scored on (among other things):
- torso height never dropping below a small clearance,
- bounded torso pitch envelope and horizontal drift,
- residual pose and drift at the end of each push case,
- bounded joint velocity (no oscillation/jitter),
- finite states throughout (no NaN / solver blowup),
- and the feedback-sensitivity / sign probes.

See the README for the full rubric.

## Constraints

- Do **not** rely on randomness — the grader uses a fixed seed and pinned
  physics settings, and your policy must too.
- Do **not** read or write files outside `/tmp/output`.
- Do **not** assume a single rollout condition; the grader runs all four
  scenarios in independent processes with reset state.
- The biped model is fixed (`/data/planar_biped.xml`). You cannot change
  its morphology, masses, joint ranges, friction, or actuators.
