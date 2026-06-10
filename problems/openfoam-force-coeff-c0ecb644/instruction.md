# OpenFOAM Engineering Repair Ticket

You are given a public starter OpenFOAM case at `/data/starter_case.tar.gz`.

The case belongs to the `force_coeff_instrumentation` workflow. Your job is to instrument deterministic force coefficients while preserving the flow setup and mesh.

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

## Engineering evidence expected

Your fix should leave a case that can be meshed, checked, and solved normally.
Where the workflow involves instrumentation or calibrated behavior, preserve
OpenFOAM function objects and outputs that demonstrate residual convergence,
bounded continuity error, and relevant engineering responses such as flow rate,
pressure drop, source response, force coefficients, scalar bounds, or mesh
quality. Hidden validation checks these kinds of evidence without exposing the
private operating points.
