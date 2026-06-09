# Underwater Glider Thermocline Tracking

Create `/tmp/output/policy.py` containing a deterministic policy for the provided MuJoCo glider environment. The policy must expose either `act(obs)`, `get_action(obs)`, or `Policy.act(obs)` and return `[pitch_command, buoyancy_command]`, with each command in `[-1, 1]`.

The glider travels in an `x`/`z` transect through a hidden thermocline. It cannot stop and reverse, so successful policies need to anticipate depth changes, compensate for the observed current, and scan sample windows in order before reaching the finish. A sample is credited only after a short continuous dwell inside the active window, so merely clipping the edge of a sample is not enough for full credit.

Each observation includes position, velocity, pitch, local current, the current thermocline depth and slope, the active sample or finish goal, sample dwell progress, workspace bounds, and plume regions. Hidden scenarios perturb currents, eddies, plume locations, and thermocline shape; some hidden internal-wave surveys place narrow sample windows next to plume corridors, so a controller must trade off dwell accuracy with real clearance instead of simply clamping to the sample center.

Your score rewards:

- scanning the thermocline sample windows in order with continuous in-window dwell;
- tracking the hidden thermocline curve across the transect;
- finishing near the requested endpoint after all samples;
- avoiding workspace violations and plume regions;
- using smooth, moderate commands across hidden scenarios.

The tracking metric gives full credit near mean depth error `<=0.050` and
p90 depth error `<=0.095`, with little credit by mean error `>=0.26` or
p90 error `>=0.34`. Finish quality is measured over the final window with
full credit near distance `<=0.070`, depth error `<=0.040`, and
current-relative drift `<=0.045`.

The headline score is gated by the weakest hidden survey's ordered-sample
completion and finish quality. Per-survey sample, tracking, and finish credit
are also safety-gated: safety below `0.20` zeros those per-survey credits and
the safety gate reaches full strength at `0.78`. The headline completion gate
is zero below weakest hidden ordered-sample score `0.70` and reaches full
strength at `0.90`; it is also zero below weakest hidden finish score `0.20`
and reaches full strength at `0.50`. A policy that tracks the thermocline well
on average but clips a plume, leaves samples unscanned, or fails to finish one
hidden survey will receive low credit.
