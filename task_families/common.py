from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

from synthetic_openfoam_pipeline.openfoam.dictionary_edit import replace_required
from synthetic_openfoam_pipeline.openfoam.case_model import pick_seed_for_family
from synthetic_openfoam_pipeline.schemas import (
    CorruptionSpec,
    DictionaryEditSpec,
    ExpectedMetrics,
    ForbiddenEditSpec,
    HiddenVariant,
    OpenFoamTaskSpec,
    OracleRepairSpec,
)


@dataclass(frozen=True)
class FamilyDefinition:
    name: str
    title: str
    symptom: str
    prompt: str
    rationale: str
    task_spec: OpenFoamTaskSpec

    @property
    def hidden_variants(self) -> list[HiddenVariant]:
        return self.task_spec.hidden_variant_grid

    @property
    def expected_metrics(self) -> ExpectedMetrics:
        return self.task_spec.expected_metrics

    @property
    def forbidden_paths(self) -> list[str]:
        return [edit.path for edit in self.task_spec.forbidden_edits]


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
    _ = task_id
    common_forbidden = [_forbidden("constant/transportProperties")]
    if name == "boundary_condition_repair":
        corruption = _edit("0/p", "outlet { type fixedValue; value uniform 0; }", "outlet { type zeroGradient; }", "break outlet pressure anchoring")
        return _family(
            name=name,
            title="Boundary-condition repair",
            symptom="The channel case runs unreliably and violates pressure/flux consistency at hidden operating points.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore fixed outlet pressure")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "fluxBalance"]),
            forbidden_edits=common_forbidden,
            prompt=_prompt(name, "repair the pressure/velocity boundary-condition consistency without changing viscosity"),
            rationale="Likely Claude-hard because the visible symptom can be patched several ways, but hidden flow reversal and viscosity variants require multi-file OpenFOAM BC consistency rather than a single text edit.",
        )
    if name == "postprocessing_instrumentation":
        corruption = _edit("system/controlDict", "functions\n{", "functions {}\n", "remove deterministic function objects", mode="replace_tail")
        return _family(
            name=name,
            title="Post-processing instrumentation",
            symptom="The case solves but produces no residual, probe, flux, or field extrema evidence for review.",
            corruption=[corruption],
            oracle_repair=[_edit("system/controlDict", "functions {}", _function_objects(), "restore deterministic function objects")],
            hidden_variant_grid=_variants("write_interval", "probe_location"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "probes", "fieldMinMax"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p")],
            prompt=_prompt(name, "instrument the case with deterministic function objects while preserving the physics"),
            rationale="Likely Claude-hard because the prompt asks for evidence generation, not a named function-object list, and hidden probes/write intervals reject brittle one-point instrumentation.",
        )
    if name == "solver_convergence_repair":
        corruption = _edit("system/fvSchemes", "div(phi,U) Gauss linearUpwind grad(U);", "div(phi,U) Gauss linear;", "destabilize convection scheme")
        return _family(
            name=name,
            title="Solver convergence repair",
            symptom="The case diverges under the fixed verifier budget after a scheme/tolerance change.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore bounded convection scheme")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U")],
            prompt=_prompt(name, "repair numerics and relaxation so the case converges without changing physical inputs"),
            rationale="Likely Claude-hard because stable OpenFOAM convergence requires coordinated fvSchemes/fvSolution choices, while hidden Reynolds variants expose overfit tolerances.",
        )
    if name == "mesh_quality_repair":
        corruption = _edit("system/blockMeshDict", "simpleGrading (1.5 1 1)", "simpleGrading (20 1 1)", "introduce excessive block grading")
        return _family(
            name=name,
            title="Mesh-quality repair",
            symptom="checkMesh reports avoidable quality issues from a subtle block grading problem.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore mild block grading")],
            hidden_variant_grid=_variants("mesh_grading", "inlet_velocity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["checkMesh"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p")],
            prompt=_prompt(name, "fix mesh quality while keeping the same topology and comparable cell count"),
            rationale="Likely Claude-hard because the repair must improve mesh quality without rewriting topology or changing physics; hidden grading checks punish broad rewrites.",
        )
    if name == "source_term_calibration":
        corruption = _edit("constant/fvOptions", "U ((0.200 0 0) 0);", "U ((0.000 0 0) 0);", "zero out calibrated source term")
        return _family(
            name=name,
            title="Source-term calibration",
            symptom="The scalar/momentum source is present but calibrated to the wrong target response.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore calibrated source term")],
            hidden_variant_grid=_variants("source_strength", "viscosity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["fieldAverage"], source_target=0.2, tolerance=0.05),
            forbidden_edits=common_forbidden + [_forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "calibrate the source behavior to hit the engineering target across hidden variants"),
            rationale="Likely Claude-hard because direct metric spoofing is rejected and the correction must generalize to hidden source-strength and viscosity variants.",
        )
    raise ValueError(f"Unknown task family: {name}")


def _family(
    *,
    name: str,
    title: str,
    symptom: str,
    corruption: list[DictionaryEditSpec],
    oracle_repair: list[DictionaryEditSpec],
    hidden_variant_grid: list[HiddenVariant],
    expected_metrics: ExpectedMetrics,
    forbidden_edits: list[ForbiddenEditSpec],
    prompt: str,
    rationale: str,
) -> FamilyDefinition:
    task_spec = OpenFoamTaskSpec(
        family=name,
        seed_case=pick_seed_for_family(name),
        corruption=CorruptionSpec(corruption, "Public starter-case perturbations"),
        oracle_repair=OracleRepairSpec(oracle_repair, "Minimal repair expected from the oracle solution"),
        hidden_variant_grid=hidden_variant_grid,
        expected_metrics=expected_metrics,
        forbidden_edits=forbidden_edits,
    )
    return FamilyDefinition(
        name=name,
        title=title,
        symptom=symptom,
        prompt=prompt,
        rationale=rationale,
        task_spec=task_spec,
    )


def _edit(path: str, old: str, new: str, description: str, *, mode: str = "replace") -> DictionaryEditSpec:
    return DictionaryEditSpec(path=path, old=old, new=new, description=description, mode=mode)


def _reverse(edit: DictionaryEditSpec, description: str) -> DictionaryEditSpec:
    return _edit(edit.path, edit.new, edit.old, description)


def _forbidden(path: str) -> ForbiddenEditSpec:
    return ForbiddenEditSpec(path=path)


def _function_objects() -> str:
    return """functions
{
    residuals
    {
        type residuals;
        libs ("libutilityFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }
    probes
    {
        type probes;
        libs ("libsampling.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
        probeLocations ((1 0.5 0.05));
    }
    fieldMinMax
    {
        type fieldMinMax;
        libs ("libfieldFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }
}
"""


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
    for step in definition.task_spec.corruption.steps:
        _apply_dictionary_edit(case_dir / step.path, step)
    marker_dir = case_dir / ".synthetic_openfoam"
    if marker_dir.exists():
        import shutil

        shutil.rmtree(marker_dir)


def _apply_dictionary_edit(path: Path, step: DictionaryEditSpec) -> None:
    if step.mode == "replace":
        replace_required(path, step.old, step.new)
        return
    if step.mode == "replace_tail":
        text = path.read_text()
        if step.old not in text:
            raise ValueError(f"Expected function-object block in {path}")
        path.write_text(text.split(step.old, 1)[0] + step.new)
        return
    raise ValueError(f"Unknown dictionary edit mode: {step.mode}")


def apply_oracle_marker(case_dir: Path, definition: FamilyDefinition) -> None:
    _ = definition
    marker_dir = case_dir / ".synthetic_openfoam"
    if marker_dir.exists():
        import shutil

        shutil.rmtree(marker_dir)

