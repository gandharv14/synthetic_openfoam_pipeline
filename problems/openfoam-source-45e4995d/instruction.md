# OpenFOAM Engineering Repair Ticket

You are given a public starter OpenFOAM case at `/data/starter_case.tar.gz`.

The case belongs to the `source_term_calibration` workflow. Your job is to calibrate the source behavior to hit the engineering target across hidden variants.

## Required submission

Write exactly this final artifact:

```text
/tmp/output/fixed_case.tar.gz
```

The archive must contain a corrected OpenFOAM case directory with the usual `0/`,
`constant/`, and `system/` folders. Only outputs under `/tmp/output` are graded.

## Constraints

- Keep the case small and deterministic.
- Do not change physical constants unless the ticket explicitly requires it.
- Do not replace the OpenFOAM case with a synthetic metric file; the grader extracts
  and validates the case structure, hidden variants, logs, and metrics.
- The hidden tests perturb operating conditions such as inlet speed, viscosity,
  write interval, probe location, source strength, or mild mesh grading.
- Preserve engineering intent rather than tuning to a single visible run.
