from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from synthetic_openfoam_pipeline.openfoam.dictionary_edit import replace_required
from synthetic_openfoam_pipeline.schemas import ExpectedMetrics, HiddenVariant


@dataclass(frozen=True)
class FamilyDefinition:
    name: str
    title: str
    symptom: str
    starter_edit: tuple[str, str, str] | None
    hidden_variants: list[HiddenVariant]
    expected_metrics: ExpectedMetrics
    forbidden_paths: list[str]
    prompt: str
    rationale: str


def _variants(*names: str) -> list[HiddenVariant]:
    lookup = {
        "inlet_velocity": HiddenVariant("changed_inlet_velocity", {"inlet_velocity": 1.25}),
        "viscosity": HiddenVariant("changed_viscosity", {"viscosity": 0.0125}),
        "write_interval": HiddenVariant("changed_write_interval", {"write_interval": 7}),
        "probe_location": HiddenVariant("changed_probe_location", {"probe_location": [0.75, 0.1, 0.0]}),
        "source_strength": HiddenVariant("changed_source_strength", {"source_strength": 0.24}),
        "mesh_grading": HiddenVariant("mild_mesh_grading", {"mesh_grading": 1.8}),
    }
    return [lookup[name] for name in names]


def family_definition(name: str, task_id: str) -> FamilyDefinition:
    common_forbidden = ["constant/transportProperties"]
    if name == "boundary_condition_repair":
        return FamilyDefinition(
            name=name,
            title="Boundary-condition repair",
            symptom="The channel case runs unreliably and violates pressure/flux consistency at hidden operating points.",
            starter_edit=("0/p", "outlet { type fixedValue; value uniform 0; }", "outlet { type zeroGradient; }"),
            hidden_variants=_variants("inlet_velocity", "viscosity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "fluxBalance"]),
            forbidden_paths=common_forbidden,
            prompt=_prompt(name, "repair the pressure/velocity boundary-condition consistency without changing viscosity"),
            rationale="Likely Claude-hard because the visible symptom can be patched several ways, but hidden flow reversal and viscosity variants require multi-file OpenFOAM BC consistency rather than a single text edit.",
        )
    if name == "postprocessing_instrumentation":
        return FamilyDefinition(
            name=name,
            title="Post-processing instrumentation",
            symptom="The case solves but produces no residual, probe, flux, or field extrema evidence for review.",
            starter_edit=None,
            hidden_variants=_variants("write_interval", "probe_location"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "probes", "fieldMinMax"]),
            forbidden_paths=common_forbidden + ["0/U", "0/p"],
            prompt=_prompt(name, "instrument the case with deterministic function objects while preserving the physics"),
            rationale="Likely Claude-hard because the prompt asks for evidence generation, not a named function-object list, and hidden probes/write intervals reject brittle one-point instrumentation.",
        )
    if name == "solver_convergence_repair":
        return FamilyDefinition(
            name=name,
            title="Solver convergence repair",
            symptom="The case diverges under the fixed verifier budget after a scheme/tolerance change.",
            starter_edit=("system/fvSchemes", "div(phi,U) Gauss linearUpwind grad(U);", "div(phi,U) Gauss linear;"),
            hidden_variants=_variants("inlet_velocity", "viscosity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals"]),
            forbidden_paths=common_forbidden + ["0/U"],
            prompt=_prompt(name, "repair numerics and relaxation so the case converges without changing physical inputs"),
            rationale="Likely Claude-hard because stable OpenFOAM convergence requires coordinated fvSchemes/fvSolution choices, while hidden Reynolds variants expose overfit tolerances.",
        )
    if name == "mesh_quality_repair":
        return FamilyDefinition(
            name=name,
            title="Mesh-quality repair",
            symptom="checkMesh reports avoidable quality issues from a subtle block grading problem.",
            starter_edit=("system/blockMeshDict", "simpleGrading (1.5 1 1)", "simpleGrading (20 1 1)"),
            hidden_variants=_variants("mesh_grading", "inlet_velocity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["checkMesh"]),
            forbidden_paths=common_forbidden + ["0/U", "0/p"],
            prompt=_prompt(name, "fix mesh quality while keeping the same topology and comparable cell count"),
            rationale="Likely Claude-hard because the repair must improve mesh quality without rewriting topology or changing physics; hidden grading checks punish broad rewrites.",
        )
    if name == "source_term_calibration":
        return FamilyDefinition(
            name=name,
            title="Source-term calibration",
            symptom="The scalar/momentum source is present but calibrated to the wrong target response.",
            starter_edit=("constant/fvOptions", "U ((0.200 0 0) 0);", "U ((0.000 0 0) 0);"),
            hidden_variants=_variants("source_strength", "viscosity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["fieldAverage"], source_target=0.2, tolerance=0.05),
            forbidden_paths=common_forbidden + ["system/blockMeshDict"],
            prompt=_prompt(name, "calibrate the source behavior to hit the engineering target across hidden variants"),
            rationale="Likely Claude-hard because direct metric spoofing is rejected and the correction must generalize to hidden source-strength and viscosity variants.",
        )
    raise ValueError(f"Unknown task family: {name}")


def _prompt(name: str, objective: str) -> str:
    return f"""# OpenFOAM Engineering Repair Ticket

You are given a public starter OpenFOAM case at `/data/starter_case.tar.gz`.

The case belongs to the `{name}` workflow. Your job is to {objective}.

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
"""


def apply_starter_perturbation(case_dir: Path, definition: FamilyDefinition) -> None:
    if definition.name == "postprocessing_instrumentation":
        path = case_dir / "system" / "controlDict"
        text = path.read_text()
        marker = "functions\n{"
        if marker not in text:
            raise ValueError(f"Expected function-object block in {path}")
        path.write_text(text.split(marker, 1)[0] + "functions {}\n")
    elif definition.starter_edit:
        rel, old, new = definition.starter_edit
        path = case_dir / rel
        replace_required(path, old, new)
    marker_dir = case_dir / ".synthetic_openfoam"
    if marker_dir.exists():
        import shutil

        shutil.rmtree(marker_dir)


def apply_oracle_marker(case_dir: Path, definition: FamilyDefinition) -> None:
    _ = definition
    marker_dir = case_dir / ".synthetic_openfoam"
    if marker_dir.exists():
        import shutil

        shutil.rmtree(marker_dir)

