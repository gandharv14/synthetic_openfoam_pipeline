from __future__ import annotations
import json
import math
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

from synthetic_openfoam_pipeline.openfoam.dictionary_edit import apply_structured_edit, replace_required
from synthetic_openfoam_pipeline.openfoam.case_model import (
    SCALAR_AMBIENT_VALUE,
    SCALAR_HOT_VALUE,
    SNAPPY_LOCATION_IN_MESH,
    SNAPPY_REFINEMENT_LEVEL,
    TRANSIENT_AVERAGE_WINDOW_START,
    TRANSIENT_DELTA_T,
    TRANSIENT_END_TIME,
    TRANSIENT_PIMPLE_BLOCK,
    case_geometry_stats,
    pick_seed_for_family,
    turbulence_inlet_patch_line,
    turbulence_inlet_values,
    turbulence_second_field,
    turbulence_wall_patch_line,
)
from synthetic_openfoam_pipeline.utils.hashing import task_id_for
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

DIFFICULTY_CORRUPTION_COUNTS = {"easy": 1, "medium": 2, "hard": 3}

# Case files each family's corruption touches. Used to keep chained
# corruptions on disjoint targets so every repair stays independently
# invertible.
CORRUPTION_TARGET_PATHS: dict[str, tuple[str, ...]] = {
    "boundary_condition_repair": ("0/p",),
    "postprocessing_instrumentation": ("system/controlDict",),
    "solver_convergence_repair": ("system/fvSchemes",),
    "mesh_quality_repair": ("system/blockMeshDict",),
    "source_term_calibration": ("constant/fvOptions",),
    "flow_rate_targeting": ("0/U",),
    "pressure_drop_calibration": ("0/p",),
    # The hash-picked corruption mode touches exactly one of these (steady ddt
    # scheme, Courant-violating deltaT, or broken PIMPLE loop); the path set is
    # the union across modes so secondary stacking stays disjoint.
    "transient_setup_repair": ("system/controlDict", "system/fvSchemes", "system/fvSolution"),
    "force_coeff_instrumentation": ("system/controlDict",),
    "scalar_transport_repair": ("0/T",),
    "mesh_resolution_tradeoff": ("system/blockMeshDict",),
    "relaxation_factor_repair": ("system/fvSolution",),
    "wall_patch_type_repair": ("0/U",),
    "field_dimensions_repair": ("0/p",),
    "residual_control_repair": ("system/fvSolution",),
    "probe_placement_repair": ("system/controlDict",),
    # The model-specific corruption touches 0/k plus the seed's dissipation
    # field (0/omega for kOmegaSST, 0/epsilon for kEpsilon); the path set is
    # the union across models so secondary stacking stays disjoint.
    "turbulence_bc_repair": ("0/k", "0/omega", "0/epsilon"),
    "snappy_hex_repair": ("system/snappyHexMeshDict",),
}

_CHANNEL_KINDS = frozenset(
    {
        "channel",
        "source_channel",
        "porous_pressure_drop_channel",
        "backward_facing_step",
        "diffuser_channel_expansion",
        "obstacle_wake",
        "transient_obstacle_wake",
        "pipe_bend",
    }
)

# Seed geometries on which a family's corruption target actually exists.
# Families absent from this map apply to every seed kind.
SECONDARY_CASE_KINDS: dict[str, frozenset[str]] = {
    "boundary_condition_repair": _CHANNEL_KINDS,
    "flow_rate_targeting": _CHANNEL_KINDS,
    "pressure_drop_calibration": _CHANNEL_KINDS,
    "source_term_calibration": frozenset({"source_channel", "porous_pressure_drop_channel"}),
    "scalar_transport_repair": frozenset({"heated_cavity"}),
    "turbulence_bc_repair": frozenset({"turbulent_channel"}),
    # The transient corruption strings only exist on the pimpleFoam seed.
    "transient_setup_repair": frozenset({"transient_obstacle_wake"}),
    # snappyHexMeshDict only exists on the 3D snappy seed.
    "snappy_hex_repair": frozenset({"snappy_obstacle_wake"}),
}

# Curated compatibility matrix: which secondary corruptions may be stacked on
# a primary family. Entries never touch the primary family's forbidden-edit
# paths or its own corruption target, so the combined corruption stays
# exactly invertible and the forbidden-edit hashes stay valid.
SECONDARY_CORRUPTION_MATRIX: dict[str, tuple[str, ...]] = {
    "boundary_condition_repair": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
        "mesh_quality_repair",
        "flow_rate_targeting",
        "source_term_calibration",
    ),
    "postprocessing_instrumentation": (
        "solver_convergence_repair",
        "mesh_quality_repair",
        "scalar_transport_repair",
    ),
    "solver_convergence_repair": (
        "boundary_condition_repair",
        "postprocessing_instrumentation",
        "mesh_quality_repair",
        "source_term_calibration",
    ),
    "mesh_quality_repair": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
        "scalar_transport_repair",
    ),
    "source_term_calibration": (
        "solver_convergence_repair",
        "boundary_condition_repair",
        "flow_rate_targeting",
    ),
    "flow_rate_targeting": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
        "source_term_calibration",
    ),
    "pressure_drop_calibration": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
        "source_term_calibration",
    ),
    # The transient seed's controlDict/fvSchemes/fvSolution all belong to the
    # primary corruption, so stack only field-file corruptions on top of it.
    "transient_setup_repair": (
        "boundary_condition_repair",
        "wall_patch_type_repair",
        "field_dimensions_repair",
    ),
    "force_coeff_instrumentation": (
        "solver_convergence_repair",
        "scalar_transport_repair",
    ),
    # postprocessing_instrumentation must not stack here: its oracle restores
    # the standard functions block, which would strip the scalarTransport
    # functionObject the scalar grading depends on.
    "scalar_transport_repair": (
        "solver_convergence_repair",
    ),
    "mesh_resolution_tradeoff": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
    ),
    "relaxation_factor_repair": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
        "mesh_quality_repair",
        "source_term_calibration",
    ),
    "wall_patch_type_repair": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
        "source_term_calibration",
    ),
    "field_dimensions_repair": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
        "source_term_calibration",
    ),
    "residual_control_repair": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
    ),
    "probe_placement_repair": (
        "solver_convergence_repair",
        "source_term_calibration",
    ),
    "turbulence_bc_repair": (
        "solver_convergence_repair",
        "postprocessing_instrumentation",
    ),
    # postprocessing_instrumentation must not stack here: its oracle restores
    # the standard functions block, which would drop the obstacle forceCoeffs
    # the snappy grading depends on.
    "snappy_hex_repair": (
        "solver_convergence_repair",
    ),
}


def select_secondary_corruption_families(
    primary: str,
    case_kind: str,
    *,
    seed: int,
    index: int,
    count: int,
) -> list[str]:
    """Hash-select up to `count` stackable secondary corruption families.

    Selection is pure stable-hash ordering over (family, seed, index) - no
    RNG - filtered to corruptions that exist on the chosen seed geometry and
    that target case files disjoint from every other applied corruption.
    """
    if count <= 0:
        return []
    pool = [
        name
        for name in SECONDARY_CORRUPTION_MATRIX.get(primary, ())
        if SECONDARY_CASE_KINDS.get(name) is None or case_kind in SECONDARY_CASE_KINDS[name]
    ]
    order = sorted(
        pool,
        key=lambda name: stable_hash({"primary": primary, "seed": seed, "index": index, "secondary": name}, 16),
    )
    selected: list[str] = []
    used_paths = set(CORRUPTION_TARGET_PATHS[primary])
    for name in order:
        paths = set(CORRUPTION_TARGET_PATHS[name])
        if used_paths & paths:
            continue
        selected.append(name)
        used_paths |= paths
        if len(selected) == count:
            break
    return selected


# Achievable goal-seeking target grids for the calibration families. The
# target is hash-picked FIRST; the oracle dictionary value is derived from it
# and the starter is corrupted away from it.
GOAL_TARGET_GRIDS: dict[str, dict[str, tuple[float, ...]]] = {
    "flow_rate_targeting": {"inlet_velocity": (0.8, 1.0, 1.25, 1.5)},
    "source_term_calibration": {"source_strength": (0.18, 0.2, 0.22)},
    "pressure_drop_calibration": {"outlet_pressure": (0.0, 0.5, 1.0)},
}


def goal_target(family: str, selector: Any, name: str) -> float:
    """Hash-pick a physical target from the curated achievable grid (no RNG)."""
    values = GOAL_TARGET_GRIDS[family][name]
    digest = int(stable_hash({"family": family, "selector": selector, "target": name}, 16), 16)
    return values[digest % len(values)]


# Per-axis derivation specs for the per-task hidden-variant grids. Axis
# values are hash-derived per task (stable_hash over task id, axis, slot),
# scaled into a physically safe range and snapped to a finite step lattice,
# so the same operating points never recur across the dataset while staying
# fully reproducible. `scale` ranges multiply the task's base value (e.g. the
# goal-seeking inlet target); `range` bounds are absolute.
HIDDEN_VARIANT_SLOTS_PER_AXIS = 4

HIDDEN_VARIANT_AXIS_SPECS: dict[str, dict[str, Any]] = {
    "inlet_velocity": {"scale": (0.7, 1.7), "step": 0.05, "default_base": 1.0},
    "viscosity": {"scale": (0.65, 1.6), "step": 0.0005, "default_base": 0.01},
    "write_interval": {"range": (5, 25), "step": 1},
    # Probe must land inside every supported domain ([0,1]x[0,1] cavity is the
    # smallest); z pinned to the mid-depth plane.
    "probe_location": {"probe_range": (0.2, 0.8), "step": 0.05},
    # Stay within the grader's source tolerance band around the task target.
    "source_strength": {"scale": (0.8, 1.2), "step": 0.005, "default_base": 0.2},
    # The grader requires simpleGrading x <= 2.0.
    "mesh_grading": {"range": (1.2, 2.0), "step": 0.05},
}


def _lattice(low: float, high: float, step: float) -> list[float]:
    start = math.ceil(round(low / step, 9))
    end = math.floor(round(high / step, 9))
    return [round(k * step, 10) for k in range(start, end + 1)]


def _axis_candidates(axis: str, base: float | None) -> list[Any]:
    spec = HIDDEN_VARIANT_AXIS_SPECS[axis]
    step = float(spec["step"])
    if "probe_range" in spec:
        lattice = _lattice(spec["probe_range"][0], spec["probe_range"][1], step)
        return [[x, y, 0.05] for x in lattice for y in lattice]
    if "scale" in spec:
        scale_low, scale_high = spec["scale"]
        anchor = float(base) if base is not None else float(spec["default_base"])
        values = _lattice(anchor * scale_low, anchor * scale_high, step)
    else:
        values = _lattice(spec["range"][0], spec["range"][1], step)
    if step == 1:
        return [int(value) for value in values]
    return values


def derive_axis_values(task_key: str, axis: str, base: float | None = None, slots: int = HIDDEN_VARIANT_SLOTS_PER_AXIS) -> list[Any]:
    """Hash-derive `slots` distinct lattice values for one variant axis."""
    candidates = _axis_candidates(axis, base)
    values: list[Any] = []
    taken: set[Any] = set()
    for slot in range(slots):
        digest = int(stable_hash({"task": task_key, "axis": axis, "slot": slot}, 16), 16)
        position = digest % len(candidates)
        key = json.dumps(candidates[position])
        while key in taken:  # deterministic linear probe for distinct values
            position = (position + 1) % len(candidates)
            key = json.dumps(candidates[position])
        taken.add(key)
        values.append(candidates[position])
    return values


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


def _variants(*names: str, task_key: str, bases: dict[str, float] | None = None) -> list[HiddenVariant]:
    for name in names:
        if name not in HIDDEN_VARIANT_AXIS_SPECS:
            raise ValueError(f"Unknown hidden variant axis: {name}")
    per_axis = {name: derive_axis_values(task_key, name, (bases or {}).get(name)) for name in names}
    variants: list[HiddenVariant] = []
    for offset, values in enumerate(product(*(per_axis[name] for name in names))):
        changes = dict(zip(names, values))
        variants.append(HiddenVariant(f"grid_variant_{offset:02d}", changes))
    return variants


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


def task_key_for(name: str, seed_selector: Any) -> str:
    """The task identity used to derive per-task hidden-variant grids."""
    spec = pick_seed_for_family(name, seed_selector)
    if isinstance(seed_selector, dict) and "seed" in seed_selector and "index" in seed_selector:
        return task_id_for(int(seed_selector["seed"]), name, spec.case_kind, int(seed_selector["index"]))
    return f"{name}:{spec.case_kind}:{stable_hash({'selector': seed_selector}, 12)}"


def family_definition(name: str, task_id: str, seed_selector: Any = None, *, goal_targets: bool = True) -> FamilyDefinition:
    """Build a family definition.

    `goal_targets=False` pins the calibration families to their registry
    defaults; used when a family is stacked as a secondary corruption so its
    edits match the primary family's seed dictionaries.
    """
    task_key = task_id or task_key_for(name, seed_selector)
    common_forbidden = [_forbidden("constant/transportProperties")]
    if name == "boundary_condition_repair":
        corruption = _edit("0/p", "outlet { type fixedValue; value uniform 0; }", "outlet { type zeroGradient; }", "break outlet pressure anchoring")
        return _family(
            name=name,
            title="Boundary-condition repair",
            symptom="The channel case runs unreliably and violates pressure/flux consistency at hidden operating points.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore fixed outlet pressure")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key),
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
            hidden_variant_grid=_variants("write_interval", "probe_location", task_key=task_key),
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
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key),
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
            hidden_variant_grid=_variants("mesh_grading", "inlet_velocity", task_key=task_key),
            expected_metrics=ExpectedMetrics(postprocessing_required=["checkMesh"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p")],
            prompt=_prompt(name, "fix mesh quality while keeping the same topology and comparable cell count"),
            rationale="Likely Claude-hard because the repair must improve mesh quality without rewriting topology or changing physics; hidden grading checks punish broad rewrites.",
            seed_selector=seed_selector,
        )
    if name == "source_term_calibration":
        target = goal_target(name, seed_selector, "source_strength") if goal_targets else 0.2
        corruption = _edit(
            "constant/fvOptions",
            f"U (({target:.3f} 0 0) 0);",
            "U ((0.000 0 0) 0);",
            "zero out calibrated source term",
        )
        return _family(
            name=name,
            title="Source-term calibration",
            symptom="The scalar/momentum source is present but calibrated to the wrong target response.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore calibrated source term")],
            hidden_variant_grid=_variants("source_strength", "viscosity", task_key=task_key, bases={"source_strength": target}),
            expected_metrics=ExpectedMetrics(postprocessing_required=["fieldAverage"], source_target=target, tolerance=0.08),
            forbidden_edits=common_forbidden + [_forbidden("system/blockMeshDict")],
            prompt=_prompt(
                name,
                f"calibrate the streamwise momentum source so the case sustains a specific source strength of {target:g} m/s^2 within ±0.08",
            ),
            rationale="Likely Claude-hard because direct metric spoofing is rejected and the correction must generalize to hidden source-strength and viscosity variants.",
            seed_selector=seed_selector,
            seed_parameter_overrides={"source_strength": target},
        )
    if name == "flow_rate_targeting":
        target_velocity = goal_target(name, seed_selector, "inlet_velocity") if goal_targets else 1.0
        inlet_area = case_geometry_stats(pick_seed_for_family(name, seed_selector))["inlet_area"]
        flow_target = round(target_velocity * inlet_area, 6)
        corrupted_velocity = round(target_velocity * 0.35, 3)
        corruption = _edit(
            "0/U",
            f"inlet {{ type fixedValue; value uniform ({target_velocity:g} 0 0); }}",
            f"inlet {{ type fixedValue; value uniform ({corrupted_velocity:g} 0 0); }}",
            "miscalibrate inlet flow target",
        )
        return _family(
            name=name,
            title="Flow-rate targeting",
            symptom="The case is stable but the inlet flow rate sits far below the documented operating point.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore target inlet flow")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key, bases={"inlet_velocity": target_velocity}),
            expected_metrics=ExpectedMetrics(
                postprocessing_required=["residuals", "probes"],
                target_inlet_velocity=target_velocity,
                target_flow_rate=flow_target,
                tolerance=0.05,
            ),
            forbidden_edits=common_forbidden + [_forbidden("system/blockMeshDict"), _forbidden("0/p")],
            prompt=_prompt(
                name,
                f"adjust the case so it achieves an inlet volumetric flow rate of {flow_target:g} m^3/s within 5%, "
                "without changing viscosity, mesh topology, or outlet pressure anchoring",
            ),
            rationale="Likely Claude-hard because the visible case runs, but the agent must back out the boundary state that delivers the stated flow rate, and hidden inlet-speed and viscosity variants reject compensating elsewhere.",
            seed_selector=seed_selector,
            seed_parameter_overrides={"inlet_velocity": target_velocity},
        )
    if name == "pressure_drop_calibration":
        target_pressure = goal_target(name, seed_selector, "outlet_pressure") if goal_targets else 0.0
        corrupted_pressure = round(target_pressure + 2.0, 6)
        corruption = _edit(
            "0/p",
            f"outlet {{ type fixedValue; value uniform {target_pressure:g}; }}",
            f"outlet {{ type fixedValue; value uniform {corrupted_pressure:g}; }}",
            "shift outlet reference pressure",
        )
        return _family(
            name=name,
            title="Pressure-drop calibration",
            symptom="The channel reports a biased pressure drop because the outlet pressure datum was shifted.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore outlet pressure datum")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key),
            expected_metrics=ExpectedMetrics(
                pressure_drop_min=0.05,
                pressure_drop_max=5.0,
                postprocessing_required=["residuals", "probes"],
                outlet_pressure_target=target_pressure,
            ),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(
                name,
                f"recalibrate the case so the measured outlet gauge-pressure datum sits at {target_pressure:g} (within ±0.25), "
                "preserving the physical inlet velocity and mesh",
            ),
            rationale="Likely Claude-hard because the solver can still converge with a shifted datum, but the measured datum and hidden pressure-drop checks require the exact reference-pressure convention.",
            seed_selector=seed_selector,
            seed_parameter_overrides={"outlet_pressure": target_pressure},
        )
    if name == "transient_setup_repair":
        # The corruption flavor is hash-picked per task (no RNG): a steady
        # ddt scheme left in the transient case, a Courant-violating deltaT,
        # or a broken PIMPLE pressure-correction loop. Each is one exact,
        # invertible dictionary edit.
        mode_digest = int(stable_hash({"family": name, "selector": seed_selector, "knob": "corruption_mode"}, 16), 16)
        mode = ("ddt_scheme", "delta_t", "pimple_loop")[mode_digest % 3]
        if mode == "ddt_scheme":
            corruption = _edit(
                "system/fvSchemes",
                "ddtSchemes { default backward; }",
                "ddtSchemes { default steadyState; }",
                "leave a steady-state time scheme in the transient case",
            )
        elif mode == "delta_t":
            corruption = _edit(
                "system/controlDict",
                f"deltaT {TRANSIENT_DELTA_T:g};",
                "deltaT 0.5;",
                "raise deltaT far past the Courant-stable step",
            )
        else:
            corruption = _edit(
                "system/fvSolution",
                TRANSIENT_PIMPLE_BLOCK,
                "PIMPLE { nOuterCorrectors 1; nCorrectors 0; nNonOrthogonalCorrectors 0; }",
                "break the PIMPLE pressure-correction loop",
            )
        return _family(
            name=name,
            title="Transient setup repair",
            symptom="The vortex-shedding case produces unusable time histories because the transient integration setup is inconsistent.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore the transient integration setup")],
            hidden_variant_grid=_variants("write_interval", "inlet_velocity", task_key=task_key),
            expected_metrics=ExpectedMetrics(
                residual_max=5e-2,
                postprocessing_required=["residuals", "probes", "forceCoeffs", "fieldAverage"],
                end_time=TRANSIENT_END_TIME,
                average_window_start=TRANSIENT_AVERAGE_WINDOW_START,
                max_delta_t=TRANSIENT_DELTA_T * 1.5,
                max_courant=1.5,
            ),
            forbidden_edits=common_forbidden + [_forbidden("system/blockMeshDict")],
            prompt=_prompt(
                name,
                "repair the transient run controls so the case integrates the unsteady wake accurately to the documented "
                "horizon and reports converged time-averaged forces and probe signals",
            ),
            rationale="Likely Claude-hard because the broken case still produces output; only the time-averaged forces and probe means over the fixed window expose the inaccurate integration, and hidden inlet-speed and write-cadence variants reject snapshot-level fixes.",
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
            hidden_variant_grid=_variants("write_interval", "probe_location", task_key=task_key),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "probes", "fieldMinMax", "forceCoeffs"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "instrument deterministic force coefficients while preserving the flow setup and mesh"),
            rationale="Likely Claude-hard because successful repair requires valid OpenFOAM function-object syntax, not just adding comments or post-hoc metric files.",
            seed_selector=seed_selector,
        )
    if name == "scalar_transport_repair":
        # The corruption stays the broken heated-wall T boundary with its
        # exact reverse as the oracle; only the heated patch name follows the
        # seed (movingWall on the cavity, walls on the heated channel). When
        # stacked as a secondary (goal_targets=False) it is pinned to the
        # cavity form, matching its SECONDARY_CASE_KINDS gate.
        seed_case = pick_seed_for_family(name, seed_selector)
        hot_patch = "walls" if goal_targets and seed_case.case_kind != "heated_cavity" else "movingWall"
        corruption = _edit(
            "0/T",
            f"{hot_patch} {{ type fixedValue; value uniform {SCALAR_HOT_VALUE:g}; }}",
            f"{hot_patch} {{ type zeroGradient; }}",
            "remove heated-wall scalar boundary",
        )
        return _family(
            name=name,
            title="Scalar-transport repair",
            symptom="The transported thermal scalar never heats up: the heated-wall boundary was replaced with an unconstrained gradient, so the solved T field carries no heat into the flow.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore heated-wall scalar value")],
            hidden_variant_grid=_variants("viscosity", "write_interval", task_key=task_key),
            expected_metrics=ExpectedMetrics(
                residual_max=5e-2,
                postprocessing_required=["residuals", "fieldMinMax"],
                scalar_field="T",
                scalar_ambient_value=SCALAR_AMBIENT_VALUE,
                scalar_hot_value=SCALAR_HOT_VALUE,
                scalar_hot_patch=hot_patch,
            ),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(
                name,
                "repair the scalar boundary condition so the solved temperature field actually transports heat from the "
                "heated wall, without changing the velocity, pressure, or mesh setup",
            ),
            rationale="Likely Claude-hard because the flow solve succeeds either way; only the transported outcomes - outlet mean temperature, wall heat input, and scalar bounds measured from the solved T field - expose the broken boundary, and hidden viscosity variants shift those outcomes.",
            seed_selector=seed_selector,
        )
    if name == "mesh_resolution_tradeoff":
        seed_case = pick_seed_for_family(name, seed_selector)
        nx = int(seed_case.parameters.get("cells_x", 20))
        ny = int(seed_case.parameters.get("cells_y", 10))
        compact = f"({nx} {ny} 1) simpleGrading (1.5 1 1)"
        inflated = f"({nx * 4} {ny * 4} 1) simpleGrading (1.5 1 1)"
        corruption = _edit("system/blockMeshDict", compact, inflated, "inflate mesh beyond the low-cost validation budget")
        return _family(
            name=name,
            title="Mesh-resolution tradeoff",
            symptom="The mesh was over-refined beyond the low-cost validation budget without improving topology.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore compact validation mesh")],
            hidden_variant_grid=_variants("mesh_grading", "inlet_velocity", task_key=task_key),
            expected_metrics=ExpectedMetrics(postprocessing_required=["checkMesh"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p")],
            prompt=_prompt(name, "restore a compact mesh resolution that keeps topology and grading intent while meeting validation cost constraints"),
            rationale="Likely Claude-hard because the case remains valid when over-refined, but hidden cost and grading checks require the intended compact resolution.",
            seed_selector=seed_selector,
        )
    if name == "relaxation_factor_repair":
        corruption = _edit(
            "system/fvSolution",
            "relaxationFactors { fields { p 0.3; } equations { U 0.7; } }",
            "relaxationFactors { fields { p 0.95; } equations { U 0.95; } }",
            "destabilize under-relaxation strategy",
        )
        return _family(
            name=name,
            title="Relaxation-factor repair",
            symptom="The steady run oscillates or diverges because the under-relaxation strategy was made far too aggressive.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore stable under-relaxation")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p")],
            prompt=_prompt(name, "repair the solver relaxation strategy so the steady run converges reliably without changing physical inputs"),
            rationale="Likely Claude-hard because the case files look physically correct; only the SIMPLE relaxation strategy is wrong, and hidden Reynolds variants reject marginal settings.",
            seed_selector=seed_selector,
        )
    if name == "wall_patch_type_repair":
        corruption = _edit("0/U", "walls { type noSlip; }", "walls { type slip; }", "remove viscous wall behavior")
        return _family(
            name=name,
            title="Wall patch-type repair",
            symptom="The case converges but wall friction is missing, so the reported flow and pressure response are unphysical.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore no-slip viscous walls")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals"]),
            forbidden_edits=common_forbidden + [_forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "restore physically correct viscous wall behavior without changing the inlet, outlet, or mesh"),
            rationale="Likely Claude-hard because a slip wall still converges cleanly; the defect only shows up in the wall-shear physics that hidden anchor comparisons measure.",
            seed_selector=seed_selector,
        )
    if name == "field_dimensions_repair":
        corruption = _edit(
            "0/p",
            "dimensions [0 2 -2 0 0 0 0];",
            "dimensions [0 1 -2 0 0 0 0];",
            "break kinematic pressure dimensions",
        )
        return _family(
            name=name,
            title="Field-dimensions repair",
            symptom="The solver aborts immediately with a dimensional-consistency error in the pressure equation.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore kinematic pressure dimensions")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "repair the field metadata so the pressure field is dimensionally consistent with the solver"),
            rationale="Likely Claude-hard because the failure is a one-token metadata defect that must be diagnosed from the solver's dimension-mismatch error, not from flow physics.",
            seed_selector=seed_selector,
        )
    if name == "residual_control_repair":
        corruption = _edit(
            "system/fvSolution",
            "residualControl { p 1e-4; U 1e-4; }",
            "residualControl { p 0.5; U 0.5; }",
            "loosen convergence controls so the run stops unconverged",
        )
        return _family(
            name=name,
            title="Residual-control repair",
            symptom="The run reports early convergence after a handful of iterations while residuals are still far too large.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore strict convergence controls")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "repair the convergence controls so the run only stops once the solution is genuinely converged"),
            rationale="Likely Claude-hard because the log claims convergence; the agent must notice the residual levels at the stop point contradict the convergence claim.",
            seed_selector=seed_selector,
        )
    if name == "probe_placement_repair":
        corruption = _edit(
            "system/controlDict",
            "probeLocations ((1 0.5 0.05));",
            "probeLocations ((25 5 0.05));",
            "move probes outside the flow domain",
        )
        return _family(
            name=name,
            title="Probe-placement repair",
            symptom="The probe instrumentation produces no usable samples because the probe points lie outside the mesh.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore in-domain probe locations")],
            hidden_variant_grid=_variants("write_interval", "viscosity", task_key=task_key),
            expected_metrics=ExpectedMetrics(postprocessing_required=["residuals", "probes"]),
            forbidden_edits=common_forbidden + [_forbidden("0/U"), _forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(name, "repair the probe instrumentation so it samples pressure and velocity inside the flow domain"),
            rationale="Likely Claude-hard because the case solves cleanly; the defect is in instrumentation placement, which must be checked against the actual mesh extents.",
            seed_selector=seed_selector,
        )
    if name == "turbulence_bc_repair":
        seed_case = pick_seed_for_family(name, seed_selector)
        model = str(seed_case.parameters.get("turbulence_model", "kOmegaSST"))
        second = turbulence_second_field(model)
        values = turbulence_inlet_values(seed_case.parameters)
        # The corruption flavor is hash-picked per task (no RNG): either the
        # near-wall treatment or the freestream turbulence state is broken.
        mode_digest = int(stable_hash({"family": name, "selector": seed_selector, "knob": "corruption_mode"}, 16), 16)
        mode = ("wall_function", "freestream")[mode_digest % 2]
        if mode == "wall_function":
            k_step = _edit(
                "0/k",
                turbulence_wall_patch_line("k", values["k"]),
                "walls { type fixedValue; value uniform 0; }",
                "replace the k wall function with a zero fixed value",
            )
            second_step = _edit(
                f"0/{second}",
                turbulence_wall_patch_line(second, values[second]),
                "walls { type zeroGradient; }",
                f"strip the {second} wall function",
            )
        else:
            corrupted_k = float(f"{values['k'] * 100:.6g}")
            corrupted_second = float(f"{values[second] / 100:.6g}")
            k_step = _edit(
                "0/k",
                turbulence_inlet_patch_line(values["k"]),
                turbulence_inlet_patch_line(corrupted_k),
                "inflate the freestream turbulent kinetic energy",
            )
            second_step = _edit(
                f"0/{second}",
                turbulence_inlet_patch_line(values[second]),
                turbulence_inlet_patch_line(corrupted_second),
                f"collapse the freestream {second} level",
            )
        return _family(
            name=name,
            title="Turbulence boundary-condition repair",
            symptom="The RANS case runs with the right turbulence model, but the near-wall treatment or freestream turbulence state is inconsistent with the documented operating Reynolds number.",
            corruption=[k_step, second_step],
            oracle_repair=[
                _reverse(second_step, f"restore the {second} boundary state"),
                _reverse(k_step, "restore the k boundary state"),
            ],
            hidden_variant_grid=_variants("inlet_velocity", "write_interval", task_key=task_key),
            expected_metrics=ExpectedMetrics(
                postprocessing_required=["residuals"],
                turbulence_model=model,
                turbulence_inlet_k=values["k"],
                turbulence_inlet_omega=values["omega"] if second == "omega" else None,
                turbulence_inlet_epsilon=values["epsilon"] if second == "epsilon" else None,
            ),
            forbidden_edits=common_forbidden
            + [
                _forbidden("constant/turbulenceProperties"),
                _forbidden("0/U"),
                _forbidden("0/p"),
                _forbidden("system/blockMeshDict"),
            ],
            prompt=_prompt(
                name,
                "repair the RANS turbulence boundary conditions so the case converges with a physically consistent "
                "near-wall treatment and freestream turbulence state, without changing the turbulence model, mesh, "
                "or flow boundary conditions",
            ),
            rationale="Likely Claude-hard because the solver still produces output with broken wall functions or an inflated freestream state; the agent must reconstruct the consistent k/omega (or k/epsilon) boundary set for the documented Reynolds number, and hidden inlet-speed variants reject cosmetic edits.",
            seed_selector=seed_selector,
        )
    if name == "snappy_hex_repair":
        # The corruption flavor is hash-picked per task (no RNG): either the
        # obstacle refinement level explodes past the meshing budget, or the
        # castellation keep-point is moved inside the obstacle so snappy
        # retains the wrong region. Each is one exact, invertible edit on
        # system/snappyHexMeshDict; the background blockMesh stays fixed.
        mode_digest = int(stable_hash({"family": name, "selector": seed_selector, "knob": "corruption_mode"}, 16), 16)
        mode = ("refinement_level", "castellation")[mode_digest % 2]
        if mode == "refinement_level":
            corruption = _edit(
                "system/snappyHexMeshDict",
                SNAPPY_REFINEMENT_LEVEL,
                "level (3 3);",
                "over-refine the obstacle surface past the meshing budget",
            )
        else:
            corruption = _edit(
                "system/snappyHexMeshDict",
                SNAPPY_LOCATION_IN_MESH,
                "locationInMesh (1.1 0.5 0.2);",
                "move the castellation keep-point inside the obstacle",
            )
        return _family(
            name=name,
            title="snappyHexMesh repair",
            symptom="The 3D snappyHexMesh case produces a mesh that violates the cell-count and quality budget, or keeps the wrong mesh region entirely.",
            corruption=[corruption],
            oracle_repair=[_reverse(corruption, "restore the pinned snappyHexMesh settings")],
            hidden_variant_grid=_variants("inlet_velocity", "viscosity", task_key=task_key),
            expected_metrics=ExpectedMetrics(
                residual_max=5e-2,
                postprocessing_required=["residuals", "probes", "forceCoeffs"],
                mesh_cells_min=1200,
                mesh_cells_max=20000,
                max_nonortho=70.0,
                max_skewness=4.5,
            ),
            forbidden_edits=common_forbidden
            + [_forbidden("0/U"), _forbidden("0/p"), _forbidden("system/blockMeshDict")],
            prompt=_prompt(
                name,
                "repair the automatic 3D mesh generation so snappyHexMesh produces a quality mesh of the flow region "
                "around the obstacle within the documented cell budget, without changing the background mesh or the "
                "flow boundary conditions",
            ),
            rationale="Likely Claude-hard because the meshing utility still runs either way; the defect only shows in the generated mesh's checkMesh quality bands and the integrated forces/fluxes on the carved obstacle, which hidden inlet-speed and viscosity variants re-measure.",
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
    seed_parameter_overrides: dict[str, Any] | None = None,
) -> FamilyDefinition:
    seed_case = pick_seed_for_family(name, seed_selector)
    if seed_parameter_overrides:
        seed_case = replace(seed_case, parameters={**seed_case.parameters, **seed_parameter_overrides})
    task_spec = OpenFoamTaskSpec(
        family=name,
        seed_case=seed_case,
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


def apply_corruption_steps(case_dir: Path, steps: list[DictionaryEditSpec]) -> None:
    for step in steps:
        _apply_dictionary_edit(case_dir / step.path, step)


def apply_starter_perturbation(case_dir: Path, definition: FamilyDefinition) -> None:
    apply_corruption_steps(case_dir, definition.task_spec.corruption.steps)
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

