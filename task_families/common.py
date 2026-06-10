from __future__ import annotations
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

from synthetic_openfoam_pipeline.openfoam.dictionary_edit import apply_structured_edit, replace_required
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
from synthetic_openfoam_pipeline.utils.hashing import stable_hash


PRIVATE_VARIANTS_PER_TASK = 4

VARIANT_PARAMETER_GRIDS: dict[str, tuple[Any, ...]] = {
    "inlet_velocity": (0.75, 1.15, 1.35, 1.6),
    "viscosity": (0.0065, 0.008, 0.0125, 0.016),
    "write_interval": (5, 8, 13, 21),
    "probe_location": ((0.5, 0.25, 0.05), (0.75, 0.1, 0.0), (1.25, 0.5, 0.05), (1.6, 0.75, 0.05)),
    "source_strength": (0.16, 0.18, 0.22, 0.24),
    "mesh_grading": (1.2, 1.7, 1.85, 2.0),
}


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
    for name in names:
        if name not in VARIANT_PARAMETER_GRIDS:
            raise ValueError(f"Unknown hidden variant axis: {name}")
    variants: list[HiddenVariant] = []
    for offset, values in enumerate(product(*(VARIANT_PARAMETER_GRIDS[name] for name in names))):
        changes = {name: _grid_value(name, value) for name, value in zip(names, values)}
        variants.append(HiddenVariant(f"grid_variant_{offset:02d}", changes))
    return variants


def _grid_value(name: str, value: Any) -> Any:
    if name == "probe_location":
        return list(value)
    return value


def variant_axes(variants: list[HiddenVariant]) -> list[str]:
    axes: list[str] = []
    for variant in variants:
        for axis in variant.changes:
            if axis not in axes:
                axes.append(axis)
    return axes


def select_hidden_variants(
    grid: list[HiddenVariant],
    *,
    family: str,
    seed: int,
    index: int,
    count: int = PRIVATE_VARIANTS_PER_TASK,
) -> list[HiddenVariant]:
    if not grid:
        return []
    selected_count = min(count, len(grid))
    selector = stable_hash(
        {
            "family": family,
            "seed": seed,
            "index": index,
            "axes": variant_axes(grid),
            "grid_size": len(grid),
        },
        16,
    )
    order = sorted(
        range(len(grid)),
        key=lambda pos: stable_hash({"selector": selector, "position": pos}, 16),
    )
    return [
        HiddenVariant(
            name=f"private_variant_{slot + 1:02d}",
            changes=grid[position].changes,
            expected_overrides=grid[position].expected_overrides,
        )
        for slot, position in enumerate(order[:selected_count])
    ]


def family_definition(name: str, task_id: str, seed_selector: Any = None) -> FamilyDefinition:
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
            seed_selector=seed_selector,
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
            seed_selector=seed_selector,
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
            seed_selector=seed_selector,
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
            seed_selector=seed_selector,
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
            seed_selector=seed_selector,
        )
    if name == "flow_rate_targeting":
        corruption = _edit("0/U", "inlet { type fixedValue; value uniform (1 0 0); }", "inlet { type fixedValue; value uniform (0.35 0 0); }", "miscalibrate inlet flow target")
        return _family(
            name=name,
            title="Flow-rate targeting",
            symptom="The case is stable but the inlet flow target is set far below the documented operating point.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore target inlet flow")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "probes"]),
            forbidden_edits=common_forbidden + [_forbidden("system/blockMeshDict"), _forbidden("0/p")],
            prompt=_prompt(name, "restore the deterministic inlet flow target without changing viscosity, mesh topology, or outlet pressure anchoring"),
            rationale="Likely Claude-hard because the visible case runs, but hidden inlet-speed and viscosity variants require preserving the intended flow-control boundary rather than compensating elsewhere.",
            seed_selector=seed_selector,
        )
    if name == "pressure_drop_calibration":
        corruption = _edit("0/p", "outlet { type fixedValue; value uniform 0; }", "outlet { type fixedValue; value uniform 2; }", "shift outlet reference pressure")
        return _family(
            name=name,
            title="Pressure-drop calibration",
            symptom="The channel reports a biased pressure drop because the outlet pressure datum was shifted.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore zero outlet pressure datum")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity"),
            expected_metrics=ExpectedMetrics(pressure_drop_min=0.05, pressure_drop_max=5.0, postprocessing_required=["residuals", "probes"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "recalibrate the pressure-drop datum while preserving the physical inlet velocity and mesh"),
            rationale="Likely Claude-hard because the solver can still converge with a shifted datum, but hidden pressure-drop checks require the exact reference-pressure convention.",
            seed_selector=seed_selector,
        )
    if name == "transient_setup_repair":
        end_time = _edit("system/controlDict", "endTime 100;", "endTime 10;", "truncate deterministic run horizon")
        write_interval = _edit("system/controlDict", "writeInterval 20;", "writeInterval 100;", "hide useful time history output")
        return _family(
            name=name,
            title="Transient setup repair",
            symptom="The run-control dictionary stops too early and writes too sparsely for deterministic validation.",
            corruption=[end_time, write_interval],
            oracle_repair=[_reverse(end_time, "restore full run horizon"), _reverse(write_interval, "restore useful output cadence")],
            hidden_variant_grid=_variants("write_interval", "inlet_velocity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "repair the run-control setup so the case reaches the intended deterministic horizon and output cadence"),
            rationale="Likely Claude-hard because the correction is in run-control semantics rather than solver physics, and hidden write-cadence variants reject one-off timing edits.",
            seed_selector=seed_selector,
        )
    if name == "force_coeff_instrumentation":
        corruption = _edit("system/controlDict", "functions\n{", "functions {}\n", "remove force-coefficient instrumentation", mode="replace_tail")
        return _family(
            name=name,
            title="Force-coefficient instrumentation",
            symptom="The case solves but lacks deterministic force-coefficient outputs needed for review.",
            corruption=[corruption],
            oracle_repair=[_edit("system/controlDict", "functions {}", _force_function_objects(), "restore deterministic force-coefficient function objects")],
            hidden_variant_grid=_variants("write_interval", "probe_location"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "probes", "fieldMinMax", "forceCoeffs"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "instrument deterministic force coefficients while preserving the flow setup and mesh"),
            rationale="Likely Claude-hard because successful repair requires valid OpenFOAM function-object syntax, not just adding comments or post-hoc metric files.",
            seed_selector=seed_selector,
        )
    if name == "scalar_transport_repair":
        corruption = _edit("0/T", "movingWall { type fixedValue; value uniform 310; }", "movingWall { type zeroGradient; }", "remove heated-wall scalar boundary")
        return _family(
            name=name,
            title="Scalar-transport repair",
            symptom="The thermal scalar field is present, but the heated-wall boundary was replaced with an unconstrained gradient.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore heated-wall scalar value")],
            hidden_variant_grid=_variants("viscosity", "write_interval"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "fieldMinMax"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "repair the scalar boundary condition without changing the velocity, pressure, or mesh setup"),
            rationale="Likely Claude-hard because the scalar field is ancillary to the simple flow solve, so hidden checks focus on preserving the exact thermal boundary intent.",
            seed_selector=seed_selector,
        )
    if name == "mesh_resolution_tradeoff":
        corruption = _edit("system/blockMeshDict", "(20 10 1) simpleGrading (1.5 1 1)", "(80 40 1) simpleGrading (1.5 1 1)", "inflate mesh beyond the low-cost validation budget")
        return _family(
            name=name,
            title="Mesh-resolution tradeoff",
            symptom="The mesh was over-refined beyond the low-cost validation budget without improving topology.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore compact validation mesh")],
            hidden_variant_grid=_variants("mesh_grading", "inlet_velocity"),
            expected_metrics=ExpectedMetrics(postprocessing_required=["checkMesh"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p")],
            prompt=_prompt(name, "restore a compact mesh resolution that keeps topology and grading intent while meeting validation cost constraints"),
            rationale="Likely Claude-hard because the case remains valid when over-refined, but hidden cost and grading checks require the intended compact resolution.",
            seed_selector=seed_selector,
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
    seed_selector: Any = None,
) -> FamilyDefinition:
    task_spec = OpenFoamTaskSpec(
        family=name,
        seed_case=pick_seed_for_family(name, seed_selector),
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


def _force_function_objects() -> str:
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
    forceCoeffs
    {
        type forceCoeffs;
        libs ("libforces.so");
        writeControl timeStep;
        writeInterval 1;
        patches (walls);
        rho rhoInf;
        rhoInf 1;
        CofR (0 0 0);
        liftDir (0 1 0);
        dragDir (1 0 0);
        pitchAxis (0 0 1);
        magUInf 1;
        lRef 1;
        Aref 1;
    }
}
"""


SYMPTOM_EVIDENCE_PLACEHOLDER = "<<SYMPTOM_EVIDENCE>>"


def _prompt(name: str, objective: str) -> str:
    # The family name must not appear anywhere in the public instruction: the
    # agent has to diagnose the problem from the diagnostic evidence below,
    # not pattern-match a workflow title.
    _ = name
    return f"""# OpenFOAM Engineering Repair Ticket

You are given a public starter OpenFOAM case at `/data/starter_case.tar.gz`.

Your job is to {objective}.

## Observed symptoms

Diagnostic evidence captured from the starter case as shipped:

```text
{SYMPTOM_EVIDENCE_PLACEHOLDER}
```

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
"""


def apply_starter_perturbation(case_dir: Path, definition: FamilyDefinition) -> None:
    for step in definition.task_spec.corruption.steps:
        _apply_dictionary_edit(case_dir / step.path, step)
    marker_dir = case_dir / ".synthetic_openfoam"
    if marker_dir.exists():
        import shutil

        shutil.rmtree(marker_dir)


def _apply_dictionary_edit(path: Path, step: DictionaryEditSpec) -> None:
    if apply_structured_edit(path, step.old, step.new, mode=step.mode):
        return
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

