# Roly-Poly Upright Recovery

Design a weighted roly-poly (wobble doll) and a controller that recovers from tipped initial poses and holds upright using **one** horizontal-axis torque motor at the base pivot.

Write:

```text
/tmp/output/model.xml
/tmp/output/policy.py
```

## Model — `/tmp/output/model.xml`

Your MJCF must compile and include:

- a floor contact plane,
- a `doll` body rocking on a single hinge joint named **`tilt`** with a horizontal axis (tilt in the sagittal plane),
- a heavy lower body **`weighted_base`** and a lighter upper body **`crown`** with the base at least **3×** heavier than the crown,
- center of mass of the doll subtree **below** the pivot site named **`pivot`** in the default upright pose (weighted-bottom stability),
- exactly **one** motor actuator on `tilt` (`nu == 1`) with `|ctrlrange| <= 12` N·m,
- sensors: `tilt_pos`, `tilt_vel`, and `upright_axis` (`framezaxis` on `crown`),
- `timestep <= 0.005` s and RK4 integration.

The curved rocker should contact the floor so the doll can rock passively, but hidden scenarios require active torque to settle within the hold window.

## Policy — `/tmp/output/policy.py`

Expose `act(obs)` or `Policy().act(obs)`. Return one finite scalar torque for the base motor.

The grader passes a dictionary observation:

- `time`, `duration`
- `tilt_angle`, `tilt_vel`
- `upright_z` (crown +Z dotted with world +Z)
- `floor_friction`, `mass_offset`, `damping_scale`

Hidden scenarios vary initial tilt angle, initial tilt rate, floor friction, base mass offset, and joint damping. Your policy must recover to near-upright and hold steady without NaNs or runaway spin.

Only `/tmp/output/` is graded.
