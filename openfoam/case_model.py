from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from synthetic_openfoam_pipeline.schemas import SeedCaseSpec
from synthetic_openfoam_pipeline.utils.hashing import stable_hash
from synthetic_openfoam_pipeline.utils.subprocesses import run_command


class MeshValidationError(RuntimeError):
    """Raised when a generated seed mesh fails blockMesh/checkMesh."""


# Curated geometry grids per case kind. Values are hash-selected (no RNG)
# from (family, selector, parameter) so generation stays deterministic.
GEOMETRY_PARAMETER_GRIDS: dict[str, dict[str, tuple[Any, ...]]] = {
    "channel": {
        "channel_length": (1.5, 2.0, 2.5),
        "cells_x": (16, 20, 24),
        "cells_y": (8, 10, 12),
    },
    "source_channel": {
        "channel_length": (1.5, 2.0, 2.5),
        "cells_x": (16, 20, 24),
        "cells_y": (8, 10, 12),
    },
    "porous_pressure_drop_channel": {
        "channel_length": (1.5, 2.0, 2.5),
        "cells_x": (16, 20, 24),
        "cells_y": (8, 10, 12),
    },
    "cavity": {"cells_xy": (10, 12, 14)},
    "heated_cavity": {"cells_xy": (10, 12, 14)},
    "backward_facing_step": {
        "step_height": (0.3, 0.4, 0.5),
        "channel_length": (2.5, 3.0, 3.5),
    },
    "diffuser_channel_expansion": {
        "expansion_ratio": (1.25, 1.67, 2.0),
        "channel_length": (2.5, 3.0, 3.5),
    },
    "obstacle_wake": {
        "obstacle_width": (0.15, 0.2, 0.3),
        "channel_length": (2.5, 3.0, 3.5),
    },
    "pipe_bend": {
        "inlet_height": (0.35, 0.4, 0.45),
        "outlet_height": (1.2, 1.4, 1.6),
    },
    "turbulent_channel": {
        "channel_length": (1.5, 2.0, 2.5),
        "cells_x": (16, 20, 24),
        "cells_y": (8, 10, 12),
        "reynolds_number": (5000, 10000, 20000),
    },
    "transient_obstacle_wake": {
        "obstacle_width": (0.15, 0.2, 0.3),
        "channel_length": (2.5, 3.0, 3.5),
    },
}

# Fixed transient integration setup for the pimpleFoam vortex-shedding seed.
# deltaT keeps the Courant number well below 1 on every curated geometry, and
# the time-averaging window covers the second half of the fixed horizon so
# single-core reruns reproduce the same mean quantities.
TRANSIENT_DELTA_T = 0.02
TRANSIENT_END_TIME = 4.0
TRANSIENT_WRITE_INTERVAL = 20
TRANSIENT_AVERAGE_WINDOW_START = 2.0
TRANSIENT_PIMPLE_BLOCK = "PIMPLE { nOuterCorrectors 2; nCorrectors 2; nNonOrthogonalCorrectors 0; }"

# Heated seeds solve T through a scalarTransport functionObject; these fixed
# values define the transported scalar state and its diffusivity.
SCALAR_AMBIENT_VALUE = 300.0
SCALAR_HOT_VALUE = 310.0
SCALAR_DIFFUSIVITY = 0.01

# True-3D snappyHexMesh seed: a fixed background box channel meshed with
# multiple z-cells; the obstacle is carved by snappyHexMesh from an analytic
# searchableBox. Every snappy setting below is pinned so single-core meshing
# is exactly reproducible.
SNAPPY_BACKGROUND_CELLS = (24, 10, 4)
SNAPPY_DOMAIN = (3.0, 1.0, 0.4)
SNAPPY_OBSTACLE_MIN = (1.0, 0.4, 0.1)
SNAPPY_OBSTACLE_MAX = (1.2, 0.6, 0.3)
SNAPPY_REFINEMENT_LEVEL = "level (1 1);"
SNAPPY_LOCATION_IN_MESH = "locationInMesh (0.21 0.21 0.21);"


def _snappy_hex_mesh_dict() -> str:
    return (
        _foam_header("dictionary", "snappyHexMeshDict")
        + f"""castellatedMesh true;
snap true;
addLayers false;
geometry
{{
    obstacle
    {{
        type searchableBox;
        min ({SNAPPY_OBSTACLE_MIN[0]:g} {SNAPPY_OBSTACLE_MIN[1]:g} {SNAPPY_OBSTACLE_MIN[2]:g});
        max ({SNAPPY_OBSTACLE_MAX[0]:g} {SNAPPY_OBSTACLE_MAX[1]:g} {SNAPPY_OBSTACLE_MAX[2]:g});
    }}
}}
castellatedMeshControls
{{
    maxLocalCells 100000;
    maxGlobalCells 200000;
    minRefinementCells 0;
    nCellsBetweenLevels 2;
    features ();
    refinementSurfaces
    {{
        obstacle
        {{
            {SNAPPY_REFINEMENT_LEVEL}
            patchInfo {{ type wall; }}
        }}
    }}
    resolveFeatureAngle 30;
    refinementRegions {{}}
    {SNAPPY_LOCATION_IN_MESH}
    allowFreeStandingZoneFaces true;
}}
snapControls
{{
    nSmoothPatch 3;
    tolerance 2.0;
    nSolveIter 30;
    nRelaxIter 5;
}}
addLayersControls
{{
    relativeSizes true;
    layers {{}}
    expansionRatio 1.0;
    finalLayerThickness 0.3;
    minThickness 0.1;
    nGrow 0;
    featureAngle 60;
    nRelaxIter 3;
    nSmoothSurfaceNormals 1;
    nSmoothNormals 3;
    nSmoothThickness 10;
    maxFaceThicknessRatio 0.5;
    maxThicknessToMedialRatio 0.3;
    minMedianAxisAngle 90;
    nBufferCellsNoExtrude 0;
    nLayerIter 50;
}}
meshQualityControls
{{
    maxNonOrtho 65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave 80;
    minVol 1e-13;
    minTetQuality 1e-15;
    minArea -1;
    minTwist 0.02;
    minDeterminant 0.001;
    minFaceWeight 0.05;
    minVolRatio 0.01;
    minTriangleTwist -1;
    nSmoothScale 4;
    errorReduction 0.75;
}}
mergeTolerance 1e-6;
"""
    )

# Wall treatments emitted for the turbulent RAS seeds; the grader checks the
# same names, so they are the single source of truth.
TURBULENCE_WALL_FUNCTIONS = {
    "k": "kqRWallFunction",
    "omega": "omegaWallFunction",
    "epsilon": "epsilonWallFunction",
    "nut": "nutkWallFunction",
}


def turbulence_second_field(model: str) -> str:
    """The model-specific dissipation field solved next to k."""
    return "epsilon" if "epsilon" in model.lower() else "omega"


def turbulence_inlet_values(parameters: dict[str, Any] | None) -> dict[str, float]:
    """Deterministic turbulence inlet state derived from the hash-picked Re.

    Standard fully-developed correlations (no RNG): nu = U*H/Re, turbulence
    intensity I = 0.16*Re^(-1/8), k = 1.5*(U*I)^2, mixing length l = 0.07*H,
    epsilon = Cmu^0.75 * k^1.5 / l, omega = k^0.5 / (Cmu^0.25 * l).
    """
    params = parameters or {}
    velocity = float(params.get("inlet_velocity", 1.0))
    reynolds = float(params.get("reynolds_number", 10000))
    height = 1.0
    c_mu = 0.09
    length_scale = 0.07 * height
    viscosity = velocity * height / reynolds
    intensity = 0.16 * reynolds ** (-1.0 / 8.0)
    k = 1.5 * (velocity * intensity) ** 2
    epsilon = (c_mu**0.75) * (k**1.5) / length_scale
    omega = (k**0.5) / ((c_mu**0.25) * length_scale)
    snap = lambda value: float(f"{value:.6g}")  # noqa: E731 - stable text formatting
    return {
        "reynolds": reynolds,
        "viscosity": snap(viscosity),
        "intensity": snap(intensity),
        "k": snap(k),
        "omega": snap(omega),
        "epsilon": snap(epsilon),
    }


def turbulence_inlet_patch_line(value: float) -> str:
    """Exact inlet patch text for a turbulence scalar (k/omega/epsilon)."""
    return f"inlet {{ type fixedValue; value uniform {value:g}; }}"


def turbulence_wall_patch_line(field: str, value: float) -> str:
    """Exact wall-function patch text for a turbulence scalar."""
    return f"walls {{ type {TURBULENCE_WALL_FUNCTIONS[field]}; value uniform {value:g}; }}"


@dataclass(frozen=True)
class CaseLayout:
    length: float
    vertices: str
    blocks: str
    mesh_boundaries: str
    u_boundaries: str
    p_boundaries: str
    functions: str | None = None
    source_strength: float | None = None
    extra_fields: dict[str, str] = field(default_factory=dict)
    cell_count: int = 0
    inlet_area: float = 0.1
    internal_u: str = "(1 0 0)"
    viscosity: float = 0.01
    turbulence_model: str | None = None
    # Transported scalar solved via the scalarTransport functionObject.
    scalar_field: str | None = None
    # blockMesh defaultPatch type: quasi-2D seeds use empty front/back
    # planes; true-3D seeds use symmetry planes with multiple z-cells.
    front_back_type: str = "empty"
    # Time-integration setup; defaults reproduce the steady SIMPLE seeds.
    time_scheme: str = "steadyState"
    end_time: float = 100.0
    delta_t: float = 1.0
    write_interval: int = 20
    pimple: bool = False


def seed_specs() -> list[SeedCaseSpec]:
    return [
        SeedCaseSpec(
            seed_id="laminar-channel",
            case_kind="channel",
            solver="simpleFoam",
            compatible_families=[
                "boundary_condition_repair",
                "solver_convergence_repair",
                "flow_rate_targeting",
                "pressure_drop_calibration",
                "mesh_resolution_tradeoff",
                "relaxation_factor_repair",
                "wall_patch_type_repair",
                "field_dimensions_repair",
                "residual_control_repair",
                "probe_placement_repair",
            ],
            runtime_estimate_sec=30,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01},
        ),
        SeedCaseSpec(
            seed_id="cavity-incompressible",
            case_kind="cavity",
            solver="simpleFoam",
            compatible_families=["postprocessing_instrumentation", "mesh_quality_repair", "probe_placement_repair"],
            runtime_estimate_sec=25,
            parameters={"lid_velocity": 1.0, "viscosity": 0.01},
        ),
        SeedCaseSpec(
            seed_id="heated-scalar-channel",
            case_kind="source_channel",
            solver="simpleFoam",
            compatible_families=["source_term_calibration", "scalar_transport_repair"],
            runtime_estimate_sec=35,
            parameters={"source_strength": 0.2, "viscosity": 0.01, "hot_wall_temperature": 310},
        ),
        SeedCaseSpec(
            seed_id="backward-facing-step",
            case_kind="backward_facing_step",
            solver="simpleFoam",
            compatible_families=[
                "boundary_condition_repair",
                "solver_convergence_repair",
                "mesh_quality_repair",
                "relaxation_factor_repair",
            ],
            runtime_estimate_sec=35,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01, "step_height": 0.4},
        ),
        SeedCaseSpec(
            seed_id="diffuser-channel-expansion",
            case_kind="diffuser_channel_expansion",
            solver="simpleFoam",
            compatible_families=[
                "boundary_condition_repair",
                "solver_convergence_repair",
                "mesh_quality_repair",
            ],
            runtime_estimate_sec=35,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01, "expansion_ratio": 1.67},
        ),
        SeedCaseSpec(
            seed_id="obstacle-wake",
            case_kind="obstacle_wake",
            solver="simpleFoam",
            compatible_families=[
                "boundary_condition_repair",
                "solver_convergence_repair",
                "mesh_quality_repair",
                "wall_patch_type_repair",
                "residual_control_repair",
            ],
            runtime_estimate_sec=40,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01, "obstacle_width": 0.2},
        ),
        SeedCaseSpec(
            seed_id="heated-cavity",
            case_kind="heated_cavity",
            solver="simpleFoam",
            compatible_families=[
                "postprocessing_instrumentation",
                "mesh_quality_repair",
                "force_coeff_instrumentation",
                "scalar_transport_repair",
            ],
            runtime_estimate_sec=30,
            parameters={"lid_velocity": 1.0, "viscosity": 0.01, "hot_wall_temperature": 310},
        ),
        SeedCaseSpec(
            seed_id="porous-pressure-drop-channel",
            case_kind="porous_pressure_drop_channel",
            solver="simpleFoam",
            compatible_families=[
                "boundary_condition_repair",
                "solver_convergence_repair",
                "source_term_calibration",
                "field_dimensions_repair",
            ],
            runtime_estimate_sec=35,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01, "source_strength": 0.2},
        ),
        SeedCaseSpec(
            seed_id="snappy-obstacle-wake-3d",
            case_kind="snappy_obstacle_wake",
            solver="simpleFoam",
            compatible_families=["snappy_hex_repair"],
            runtime_estimate_sec=90,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01},
        ),
        SeedCaseSpec(
            seed_id="transient-obstacle-wake",
            case_kind="transient_obstacle_wake",
            solver="pimpleFoam",
            compatible_families=["transient_setup_repair"],
            runtime_estimate_sec=60,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.002, "obstacle_width": 0.2},
        ),
        SeedCaseSpec(
            seed_id="turbulent-channel-komega",
            case_kind="turbulent_channel",
            solver="simpleFoam",
            compatible_families=["turbulence_bc_repair"],
            runtime_estimate_sec=45,
            parameters={"inlet_velocity": 1.0, "turbulence_model": "kOmegaSST"},
        ),
        SeedCaseSpec(
            seed_id="turbulent-channel-kepsilon",
            case_kind="turbulent_channel",
            solver="simpleFoam",
            compatible_families=["turbulence_bc_repair"],
            runtime_estimate_sec=45,
            parameters={"inlet_velocity": 1.0, "turbulence_model": "kEpsilon"},
        ),
        SeedCaseSpec(
            seed_id="pipe-bend",
            case_kind="pipe_bend",
            solver="simpleFoam",
            compatible_families=[
                "boundary_condition_repair",
                "solver_convergence_repair",
                "mesh_quality_repair",
                "wall_patch_type_repair",
                "residual_control_repair",
            ],
            runtime_estimate_sec=35,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01, "bend_angle_degrees": 90},
        ),
    ]


def compatible_seed_specs(family: str) -> list[SeedCaseSpec]:
    return [spec for spec in seed_specs() if family in spec.compatible_families]


def derive_geometry_parameters(case_kind: str, family: str, selector: Any) -> dict[str, Any]:
    """Hash-select geometry parameters from the curated grids (no RNG)."""
    grid = GEOMETRY_PARAMETER_GRIDS.get(case_kind, {})
    derived: dict[str, Any] = {}
    for name in sorted(grid):
        values = grid[name]
        digest = int(stable_hash({"family": family, "selector": selector, "parameter": name}, 16), 16)
        derived[name] = values[digest % len(values)]
    return derived


def pick_seed_for_family(family: str, selector: Any = None) -> SeedCaseSpec:
    specs = compatible_seed_specs(family)
    if not specs:
        specs = [seed_specs()[0]]
    if selector is None:
        spec = specs[0]
    elif isinstance(selector, int):
        spec = specs[selector % len(specs)]
    else:
        index = int(stable_hash({"family": family, "selector": selector}, 16), 16) % len(specs)
        spec = specs[index]
    parameters = dict(spec.parameters)
    parameters.update(derive_geometry_parameters(spec.case_kind, family, selector))
    return replace(spec, parameters=parameters)


def case_geometry_stats(spec: SeedCaseSpec) -> dict[str, float]:
    """Geometry-derived quantities (inlet area, cell count) for anchors."""
    layout = _case_layout(spec.case_kind, spec.parameters)
    return {"inlet_area": layout.inlet_area, "mesh_cells": float(layout.cell_count)}


def _foam_header(cls: str, obj: str) -> str:
    return f"FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"


def _function_objects(
    *,
    probes: bool = True,
    field_min_max: bool = True,
    field_average: bool = False,
    scalar_transport: bool = False,
) -> str:
    blocks = [
        """    residuals
    {
        type residuals;
        libs ("libutilityFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }"""
    ]
    if probes:
        blocks.append(
            """    probes
    {
        type probes;
        libs ("libsampling.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
        probeLocations ((1 0.5 0.05));
    }"""
        )
    if field_min_max:
        blocks.append(
            """    fieldMinMax
    {
        type fieldMinMax;
        libs ("libfieldFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }"""
        )
    if field_average:
        blocks.append(
            """    fieldAverage
    {
        type fieldAverage;
        libs ("libfieldFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields
        (
            U
            {
                mean on;
                prime2Mean off;
                base time;
            }
        );
    }"""
        )
    if scalar_transport:
        # Actively solves the T equation during the run (bounded01 is the
        # ESI-build clipping switch; harmless extra keyword on Foundation).
        blocks.append(
            f"""    scalarTransport
    {{
        type scalarTransport;
        libs ("libsolverFunctionObjects.so");
        field T;
        D {SCALAR_DIFFUSIVITY:g};
        bounded01 false;
        writeControl timeStep;
        writeInterval 1;
    }}"""
        )
    return "functions\n{\n" + "\n".join(blocks) + "\n}\n"


def _transient_function_objects(window_start: float) -> str:
    """Instrumentation for the vortex-shedding seed.

    Probes and forceCoeffs sample every time step so the grader can average
    them over the fixed [window_start, endTime] window; fieldAverage holds the
    same window via timeStart.
    """
    return f"""functions
{{
    residuals
    {{
        type residuals;
        libs ("libutilityFunctionObjects.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
    }}
    probes
    {{
        type probes;
        libs ("libsampling.so");
        writeControl timeStep;
        writeInterval 1;
        fields (p U);
        probeLocations ((2 0.5 0.05));
    }}
    forceCoeffs
    {{
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
    }}
    fieldAverage
    {{
        type fieldAverage;
        libs ("libfieldFunctionObjects.so");
        writeControl writeTime;
        timeStart {window_start:g};
        fields
        (
            U
            {{
                mean on;
                prime2Mean off;
                base time;
            }}
            p
            {{
                mean on;
                prime2Mean off;
                base time;
            }}
        );
    }}
}}
"""


def _snappy_function_objects() -> str:
    """Instrumentation for the 3D snappy seed: integrated quantities only
    (forces on the carved obstacle, probe samples, residuals)."""
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
        probeLocations ((2 0.5 0.2));
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
        patches (obstacle);
        rho rhoInf;
        rhoInf 1;
        CofR (1.1 0.5 0.2);
        liftDir (0 1 0);
        dragDir (1 0 0);
        pitchAxis (0 0 1);
        magUInf 1;
        lRef 0.2;
        Aref 0.04;
    }
}
"""


def _fv_options(source_strength: float | None = None) -> str:
    if source_strength is None:
        return _foam_header("dictionary", "fvOptions") + "\n"
    return (
        _foam_header("dictionary", "fvOptions")
        + f"""momentumSource
{{
    type vectorSemiImplicitSource;
    active true;
    selectionMode all;
    volumeMode specific;
    injectionRateSuSp
    {{
        U (({source_strength:.3f} 0 0) 0);
    }}
}}
"""
    )


def _vol_scalar_field(obj: str, dimensions: str, internal: str, boundaries: str) -> str:
    return (
        _foam_header("volScalarField", obj)
        + f"""dimensions {dimensions};
internalField {internal};
boundaryField
{{
{boundaries}
}}
"""
    )


def _default_channel_u_boundaries(inlet_velocity: float = 1.0) -> str:
    vector = f"({inlet_velocity:g} 0 0)"
    return f"""    inlet {{ type fixedValue; value uniform {vector}; }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform {vector}; }}
    walls {{ type noSlip; }}
    frontAndBack {{ type empty; }}"""


def _default_channel_p_boundaries(outlet_pressure: float = 0.0) -> str:
    return f"""    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform {outlet_pressure:g}; }}
    walls {{ type zeroGradient; }}
    frontAndBack {{ type empty; }}"""


def _turbulence_scalar_boundaries(field: str, value: float) -> str:
    return f"""    {turbulence_inlet_patch_line(value)}
    outlet {{ type inletOutlet; inletValue uniform {value:g}; value uniform {value:g}; }}
    {turbulence_wall_patch_line(field, value)}
    frontAndBack {{ type empty; }}"""


def _turbulence_nut_boundaries() -> str:
    return f"""    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    walls {{ type {TURBULENCE_WALL_FUNCTIONS["nut"]}; value uniform 0; }}
    frontAndBack {{ type empty; }}"""


def _default_cavity_u_boundaries() -> str:
    return """    movingWall { type fixedValue; value uniform (1 0 0); }
    walls { type noSlip; }
    frontAndBack { type empty; }"""


def _default_cavity_p_boundaries() -> str:
    return """    movingWall { type zeroGradient; }
    walls { type zeroGradient; }
    frontAndBack { type empty; }"""


def _rect_vertices(length: float, height: float = 1.0, depth: float = 0.1) -> str:
    return (
        f"(0 0 0) ({length:g} 0 0) ({length:g} {height:g} 0) (0 {height:g} 0) "
        f"(0 0 {depth:g}) ({length:g} 0 {depth:g}) ({length:g} {height:g} {depth:g}) (0 {height:g} {depth:g})"
    )


def _grid_vertex_map(xs: list[float], ys: list[float], depth: float = 0.1) -> tuple[str, dict[tuple[int, int, int], int]]:
    points: list[str] = []
    indices: dict[tuple[int, int, int], int] = {}
    for k, z in enumerate((0.0, depth)):
        for j, y in enumerate(ys):
            for i, x in enumerate(xs):
                indices[(i, j, k)] = len(points)
                points.append(f"({x:g} {y:g} {z:g})")
    return " ".join(points), indices


def _block(indices: dict[tuple[int, int, int], int], i0: int, i1: int, j0: int, j1: int, cells: str) -> str:
    v = indices
    return (
        f"    hex ({v[(i0, j0, 0)]} {v[(i1, j0, 0)]} {v[(i1, j1, 0)]} {v[(i0, j1, 0)]} "
        f"{v[(i0, j0, 1)]} {v[(i1, j0, 1)]} {v[(i1, j1, 1)]} {v[(i0, j1, 1)]}) "
        f"({cells}) simpleGrading (1.5 1 1)"
    )


def _face(indices: dict[tuple[int, int, int], int], corners: tuple[tuple[int, int, int], ...]) -> str:
    return "(" + " ".join(str(indices[corner]) for corner in corners) + ")"


def _case_layout(case_kind: str, parameters: dict[str, Any] | None = None) -> CaseLayout:
    params = parameters or {}
    if case_kind == "transient_obstacle_wake":
        # Vortex shedding on the existing obstacle_wake geometry with a fixed
        # transient integration setup (backward ddt, fixed deltaT/endTime).
        base = _case_layout("obstacle_wake", params)
        return replace(
            base,
            functions=_transient_function_objects(TRANSIENT_AVERAGE_WINDOW_START),
            viscosity=float(params.get("viscosity", 0.002)),
            time_scheme="backward",
            end_time=TRANSIENT_END_TIME,
            delta_t=TRANSIENT_DELTA_T,
            write_interval=TRANSIENT_WRITE_INTERVAL,
            pimple=True,
        )
    if case_kind in {"channel", "source_channel", "porous_pressure_drop_channel"}:
        length = float(params.get("channel_length", 2.0))
        nx = int(params.get("cells_x", 20))
        ny = int(params.get("cells_y", 10))
        inlet_velocity = float(params.get("inlet_velocity", 1.0))
        outlet_pressure = float(params.get("outlet_pressure", 0.0))
        mesh_boundaries = """    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    walls { type wall; faces ((0 1 5 4) (3 7 6 2)); }"""
        source_strength = (
            float(params.get("source_strength", 0.2))
            if case_kind in {"source_channel", "porous_pressure_drop_channel"}
            else None
        )
        extra_fields: dict[str, str] = {}
        scalar_field = None
        if case_kind == "source_channel":
            # Heated channel: inlet at ambient, heated walls; the outlet uses
            # inletOutlet so the solved outlet T distribution is written and
            # the grader can measure the transported outlet mean.
            t_boundaries = f"""    inlet {{ type fixedValue; value uniform {SCALAR_AMBIENT_VALUE:g}; }}
    outlet {{ type inletOutlet; inletValue uniform {SCALAR_AMBIENT_VALUE:g}; value uniform {SCALAR_AMBIENT_VALUE:g}; }}
    walls {{ type fixedValue; value uniform {SCALAR_HOT_VALUE:g}; }}
    frontAndBack {{ type empty; }}"""
            extra_fields["0/T"] = _vol_scalar_field(
                "T", "[0 0 0 1 0 0 0]", f"uniform {SCALAR_AMBIENT_VALUE:g}", t_boundaries
            )
            scalar_field = "T"
        return CaseLayout(
            length=length,
            vertices=_rect_vertices(length),
            blocks=f"    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1.5 1 1)",
            mesh_boundaries=mesh_boundaries,
            u_boundaries=_default_channel_u_boundaries(inlet_velocity),
            p_boundaries=_default_channel_p_boundaries(outlet_pressure),
            functions=_function_objects(
                field_average=case_kind == "source_channel",
                scalar_transport=scalar_field is not None,
            ),
            source_strength=source_strength,
            extra_fields=extra_fields,
            scalar_field=scalar_field,
            cell_count=nx * ny,
            inlet_area=0.1,
            internal_u=f"({inlet_velocity:g} 0 0)",
        )
    if case_kind == "turbulent_channel":
        length = float(params.get("channel_length", 2.0))
        nx = int(params.get("cells_x", 20))
        ny = int(params.get("cells_y", 10))
        inlet_velocity = float(params.get("inlet_velocity", 1.0))
        outlet_pressure = float(params.get("outlet_pressure", 0.0))
        model = str(params.get("turbulence_model", "kOmegaSST"))
        second = turbulence_second_field(model)
        values = turbulence_inlet_values(params)
        second_dimensions = "[0 0 -1 0 0 0 0]" if second == "omega" else "[0 2 -3 0 0 0 0]"
        extra_fields = {
            "0/k": _vol_scalar_field(
                "k", "[0 2 -2 0 0 0 0]", f"uniform {values['k']:g}", _turbulence_scalar_boundaries("k", values["k"])
            ),
            f"0/{second}": _vol_scalar_field(
                second,
                second_dimensions,
                f"uniform {values[second]:g}",
                _turbulence_scalar_boundaries(second, values[second]),
            ),
            "0/nut": _vol_scalar_field("nut", "[0 2 -1 0 0 0 0]", "uniform 0", _turbulence_nut_boundaries()),
        }
        mesh_boundaries = """    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    walls { type wall; faces ((0 1 5 4) (3 7 6 2)); }"""
        return CaseLayout(
            length=length,
            vertices=_rect_vertices(length),
            blocks=f"    hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1.5 1 1)",
            mesh_boundaries=mesh_boundaries,
            u_boundaries=_default_channel_u_boundaries(inlet_velocity),
            p_boundaries=_default_channel_p_boundaries(outlet_pressure),
            functions=_function_objects(),
            extra_fields=extra_fields,
            cell_count=nx * ny,
            inlet_area=0.1,
            internal_u=f"({inlet_velocity:g} 0 0)",
            viscosity=values["viscosity"],
            turbulence_model=model,
        )
    if case_kind in {"cavity", "heated_cavity"}:
        n = int(params.get("cells_xy", 10))
        mesh_boundaries = """    movingWall { type wall; faces ((3 7 6 2)); }
    walls { type wall; faces ((0 1 5 4) (0 4 7 3) (1 2 6 5)); }"""
        extra_fields = {}
        if case_kind == "heated_cavity":
            t_boundaries = """    movingWall { type fixedValue; value uniform 310; }
    walls { type fixedValue; value uniform 300; }
    frontAndBack { type empty; }"""
            extra_fields["0/T"] = _vol_scalar_field("T", "[0 0 0 1 0 0 0]", "uniform 300", t_boundaries)
        return CaseLayout(
            length=1.0,
            vertices=_rect_vertices(1.0),
            blocks=f"    hex (0 1 2 3 4 5 6 7) ({n} {n} 1) simpleGrading (1.5 1 1)",
            mesh_boundaries=mesh_boundaries,
            u_boundaries=_default_cavity_u_boundaries(),
            p_boundaries=_default_cavity_p_boundaries(),
            functions=_function_objects(scalar_transport=case_kind == "heated_cavity"),
            extra_fields=extra_fields,
            scalar_field="T" if case_kind == "heated_cavity" else None,
            cell_count=n * n,
            inlet_area=0.1,
        )
    if case_kind == "backward_facing_step":
        step = float(params.get("step_height", 0.4))
        length = float(params.get("channel_length", 3.0))
        xs = [0.0, 1.0, length]
        ys = [0.0, step, 1.0]
        lower_y = max(int(round(10 * step)), 2)
        upper_y = max(int(round(10 * (1.0 - step))), 2)
        down_x = max(int(round(8 * (length - 1.0))), 4)
        vertices, v = _grid_vertex_map(xs, ys)
        return CaseLayout(
            length=length,
            vertices=vertices,
            blocks="\n".join(
                [
                    _block(v, 0, 1, 1, 2, f"8 {upper_y} 1"),
                    _block(v, 1, 2, 0, 1, f"{down_x} {lower_y} 1"),
                    _block(v, 1, 2, 1, 2, f"{down_x} {upper_y} 1"),
                ]
            ),
            cell_count=8 * upper_y + down_x * (lower_y + upper_y),
            inlet_area=(1.0 - step) * 0.1,
            mesh_boundaries=f"""    inlet {{ type patch; faces ({_face(v, ((0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 0)))}); }}
    outlet {{ type patch; faces ({_face(v, ((2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 0, 1)))} {_face(v, ((2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 1, 1)))}); }}
    walls {{ type wall; faces ({_face(v, ((1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)))} {_face(v, ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)))} {_face(v, ((0, 2, 0), (0, 2, 1), (1, 2, 1), (1, 2, 0)))} {_face(v, ((1, 2, 0), (1, 2, 1), (2, 2, 1), (2, 2, 0)))}); }}""",
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(),
        )
    if case_kind == "diffuser_channel_expansion":
        ratio = float(params.get("expansion_ratio", 1.67))
        length = float(params.get("channel_length", 3.0))
        opening = min(max(1.0 / ratio, 0.3), 0.9)
        y1 = round((1.0 - opening) / 2.0, 6)
        y2 = round(1.0 - y1, 6)
        mid_y = max(int(round(10 * (y2 - y1))), 3)
        outer_y = max(int(round(15 * y1)), 2)
        down_x = max(int(round(8 * (length - 1.0))), 4)
        xs = [0.0, 1.0, length]
        ys = [0.0, y1, y2, 1.0]
        vertices, v = _grid_vertex_map(xs, ys)
        return CaseLayout(
            length=length,
            vertices=vertices,
            blocks="\n".join(
                [
                    _block(v, 0, 1, 1, 2, f"8 {mid_y} 1"),
                    _block(v, 1, 2, 0, 1, f"{down_x} {outer_y} 1"),
                    _block(v, 1, 2, 1, 2, f"{down_x} {mid_y} 1"),
                    _block(v, 1, 2, 2, 3, f"{down_x} {outer_y} 1"),
                ]
            ),
            cell_count=8 * mid_y + down_x * (mid_y + 2 * outer_y),
            inlet_area=(y2 - y1) * 0.1,
            mesh_boundaries=f"""    inlet {{ type patch; faces ({_face(v, ((0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 0)))}); }}
    outlet {{ type patch; faces ({_face(v, ((2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 0, 1)))} {_face(v, ((2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 1, 1)))} {_face(v, ((2, 2, 0), (2, 3, 0), (2, 3, 1), (2, 2, 1)))}); }}
    walls {{ type wall; faces ({_face(v, ((1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)))} {_face(v, ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)))} {_face(v, ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)))} {_face(v, ((0, 2, 0), (0, 2, 1), (1, 2, 1), (1, 2, 0)))} {_face(v, ((1, 2, 0), (1, 3, 0), (1, 3, 1), (1, 2, 1)))} {_face(v, ((1, 3, 0), (1, 3, 1), (2, 3, 1), (2, 3, 0)))}); }}""",
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(),
        )
    if case_kind == "snappy_obstacle_wake":
        # True-3D obstacle wake: a fixed background box with multiple z-cells
        # and symmetry front/back planes; snappyHexMesh carves and refines the
        # obstacle from the pinned searchableBox geometry.
        length, height, depth = SNAPPY_DOMAIN
        nx, ny, nz = SNAPPY_BACKGROUND_CELLS
        inlet_velocity = float(params.get("inlet_velocity", 1.0))
        outlet_pressure = float(params.get("outlet_pressure", 0.0))
        u_boundaries = f"""    inlet {{ type fixedValue; value uniform ({inlet_velocity:g} 0 0); }}
    outlet {{ type inletOutlet; inletValue uniform (0 0 0); value uniform ({inlet_velocity:g} 0 0); }}
    walls {{ type noSlip; }}
    obstacle {{ type noSlip; }}
    frontAndBack {{ type symmetry; }}"""
        p_boundaries = f"""    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform {outlet_pressure:g}; }}
    walls {{ type zeroGradient; }}
    obstacle {{ type zeroGradient; }}
    frontAndBack {{ type symmetry; }}"""
        mesh_boundaries = """    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    walls { type wall; faces ((0 1 5 4) (3 7 6 2)); }"""
        return CaseLayout(
            length=length,
            vertices=_rect_vertices(length, height, depth),
            blocks=f"    hex (0 1 2 3 4 5 6 7) ({nx} {ny} {nz}) simpleGrading (1 1 1)",
            mesh_boundaries=mesh_boundaries,
            u_boundaries=u_boundaries,
            p_boundaries=p_boundaries,
            functions=_snappy_function_objects(),
            extra_fields={"system/snappyHexMeshDict": _snappy_hex_mesh_dict()},
            cell_count=nx * ny * nz,
            inlet_area=height * depth,
            internal_u=f"({inlet_velocity:g} 0 0)",
            front_back_type="symmetry",
        )
    if case_kind == "obstacle_wake":
        width = float(params.get("obstacle_width", 0.2))
        length = float(params.get("channel_length", 3.0))
        gap_x = max(int(round(15 * width)), 2)
        down_x = max(int(round(9 * (length - 1.0 - width))), 4)
        xs = [0.0, 1.0, round(1.0 + width, 6), length]
        ys = [0.0, 0.4, 0.6, 1.0]
        vertices, v = _grid_vertex_map(xs, ys)
        block_specs = [
            (0, 1, 0, 1, "8 4 1"),
            (0, 1, 1, 2, "8 3 1"),
            (0, 1, 2, 3, "8 4 1"),
            (1, 2, 0, 1, f"{gap_x} 4 1"),
            (1, 2, 2, 3, f"{gap_x} 4 1"),
            (2, 3, 0, 1, f"{down_x} 4 1"),
            (2, 3, 1, 2, f"{down_x} 3 1"),
            (2, 3, 2, 3, f"{down_x} 4 1"),
        ]
        return CaseLayout(
            length=length,
            vertices=vertices,
            blocks="\n".join(_block(v, *spec) for spec in block_specs),
            cell_count=(8 + down_x) * 11 + gap_x * 8,
            inlet_area=0.1,
            mesh_boundaries=f"""    inlet {{ type patch; faces ({_face(v, ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)))} {_face(v, ((0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 0)))} {_face(v, ((0, 2, 0), (0, 2, 1), (0, 3, 1), (0, 3, 0)))}); }}
    outlet {{ type patch; faces ({_face(v, ((3, 0, 0), (3, 1, 0), (3, 1, 1), (3, 0, 1)))} {_face(v, ((3, 1, 0), (3, 2, 0), (3, 2, 1), (3, 1, 1)))} {_face(v, ((3, 2, 0), (3, 3, 0), (3, 3, 1), (3, 2, 1)))}); }}
    walls {{ type wall; faces ({_face(v, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)))} {_face(v, ((1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)))} {_face(v, ((2, 0, 0), (3, 0, 0), (3, 0, 1), (2, 0, 1)))} {_face(v, ((0, 3, 0), (0, 3, 1), (1, 3, 1), (1, 3, 0)))} {_face(v, ((1, 3, 0), (1, 3, 1), (2, 3, 1), (2, 3, 0)))} {_face(v, ((2, 3, 0), (2, 3, 1), (3, 3, 1), (3, 3, 0)))} {_face(v, ((1, 1, 0), (1, 2, 0), (1, 2, 1), (1, 1, 1)))} {_face(v, ((2, 1, 0), (2, 1, 1), (2, 2, 1), (2, 2, 0)))} {_face(v, ((1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 0)))} {_face(v, ((1, 2, 0), (2, 2, 0), (2, 2, 1), (1, 2, 1)))}); }}""",
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(),
        )
    if case_kind == "pipe_bend":
        inlet_height = float(params.get("inlet_height", 0.4))
        outlet_height = float(params.get("outlet_height", 1.4))
        leg_y = max(int(round(12 * inlet_height)), 3)
        riser_y = max(int(round(12 * (outlet_height - inlet_height))), 6)
        xs = [0.0, 0.6, 1.0]
        ys = [0.0, inlet_height, outlet_height]
        vertices, v = _grid_vertex_map(xs, ys)
        return CaseLayout(
            length=outlet_height,
            vertices=vertices,
            blocks="\n".join(
                [
                    _block(v, 0, 1, 0, 1, f"8 {leg_y} 1"),
                    _block(v, 1, 2, 0, 1, f"5 {leg_y} 1"),
                    _block(v, 1, 2, 1, 2, f"5 {riser_y} 1"),
                ]
            ),
            cell_count=13 * leg_y + 5 * riser_y,
            inlet_area=inlet_height * 0.1,
            mesh_boundaries=f"""    inlet {{ type patch; faces ({_face(v, ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)))}); }}
    outlet {{ type patch; faces ({_face(v, ((1, 2, 0), (1, 2, 1), (2, 2, 1), (2, 2, 0)))}); }}
    walls {{ type wall; faces ({_face(v, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)))} {_face(v, ((1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)))} {_face(v, ((2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 0, 1)))} {_face(v, ((2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 1, 1)))} {_face(v, ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)))} {_face(v, ((1, 1, 0), (1, 2, 0), (1, 2, 1), (1, 1, 1)))}); }}""",
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(),
        )
    raise ValueError(f"Unknown seed case kind: {case_kind}")


def mesh_gate_available() -> bool:
    return shutil.which("blockMesh") is not None and shutil.which("checkMesh") is not None


def validate_seed_mesh(case_dir: Path, timeout: int = 240) -> None:
    """Gate a generated case through blockMesh (+ snappyHexMesh) + checkMesh.

    Runs on a scratch copy; cases shipping a snappyHexMeshDict are meshed
    through the full snappy pipeline before quality checking.
    """
    env = "source /etc/solver-envs.d/openfoam.sh >/dev/null 2>&1"
    tmp = Path(tempfile.mkdtemp(prefix="synthetic-openfoam-meshcheck-"))
    try:
        scratch = tmp / "case"
        shutil.copytree(case_dir, scratch)
        stages = [("blockMesh", f"blockMesh -case {scratch}")]
        if (scratch / "system" / "snappyHexMeshDict").exists():
            stages.append(("snappyHexMesh", f"snappyHexMesh -overwrite -case {scratch}"))
        stages.append(("checkMesh", f"checkMesh -case {scratch}"))
        for name, command in stages:
            result = run_command(["bash", "-lc", f"{env}; {command}"], timeout=timeout)
            log = result.stdout + result.stderr
            if not result.ok or (name == "checkMesh" and "failed" in log.lower()):
                raise MeshValidationError(
                    f"{name} rejected the generated mesh for {case_dir}:\n{log[-1500:]}"
                )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def write_seed_case(dest: Path, spec: SeedCaseSpec, *, validate_mesh: bool | None = None) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for sub in ("0", "constant", "system"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    layout = _case_layout(spec.case_kind, spec.parameters)

    (dest / "0" / "U").write_text(
        _foam_header("volVectorField", "U")
        + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform {layout.internal_u};
boundaryField
{{
{layout.u_boundaries}
}}
"""
    )
    (dest / "0" / "p").write_text(_vol_scalar_field("p", "[0 2 -2 0 0 0 0]", "uniform 0", layout.p_boundaries))
    for rel, text in layout.extra_fields.items():
        (dest / rel).write_text(text)
    (dest / "constant" / "transportProperties").write_text(
        _foam_header("dictionary", "transportProperties") + f"transportModel Newtonian;\nnu {layout.viscosity:g};\n"
    )
    if layout.turbulence_model:
        turbulence_text = f"""simulationType RAS;
RAS
{{
    RASModel {layout.turbulence_model};
    turbulence on;
    printCoeffs off;
}}
"""
    else:
        turbulence_text = "simulationType laminar;\n"
    (dest / "constant" / "turbulenceProperties").write_text(
        _foam_header("dictionary", "turbulenceProperties") + turbulence_text
    )
    (dest / "constant" / "fvOptions").write_text(_fv_options(layout.source_strength))
    (dest / "system" / "blockMeshDict").write_text(
        _foam_header("dictionary", "blockMeshDict")
        + f"""scale 1;
vertices ({layout.vertices});
blocks
(
{layout.blocks}
);
edges ();
boundary
(
{layout.mesh_boundaries}
);
defaultPatch {{ name frontAndBack; type {layout.front_back_type}; }}
mergePatchPairs ();
"""
    )
    (dest / "system" / "controlDict").write_text(
        _foam_header("dictionary", "controlDict")
        + f"""application {spec.solver};
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {layout.end_time:g};
deltaT {layout.delta_t:g};
writeControl timeStep;
writeInterval {layout.write_interval};
runTimeModifiable false;
{layout.functions or _function_objects()}"""
    )
    # Fixed schemes/solvers: turbulent seeds add upwind turbulence transport
    # and matching solver/relaxation entries; laminar text is unchanged.
    second = turbulence_second_field(layout.turbulence_model) if layout.turbulence_model else None
    turbulence_div = f" div(phi,k) Gauss upwind; div(phi,{second}) Gauss upwind;" if second else ""
    scalar_div = f" div(phi,{layout.scalar_field}) Gauss upwind;" if layout.scalar_field else ""
    wall_dist = "wallDist { method meshWave; }\n" if second else ""
    (dest / "system" / "fvSchemes").write_text(
        _foam_header("dictionary", "fvSchemes")
        + f"""ddtSchemes {{ default {layout.time_scheme}; }}
gradSchemes {{ default Gauss linear; }}
divSchemes {{ default none; div(phi,U) Gauss linearUpwind grad(U);{turbulence_div}{scalar_div} div((nuEff*dev2(T(grad(U))))) Gauss linear; }}
laplacianSchemes {{ default Gauss linear corrected; }}
interpolationSchemes {{ default linear; }}
snGradSchemes {{ default corrected; }}
{wall_dist}"""
    )
    turbulence_solvers = (
        f"""
    k {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }}
    {second} {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }}"""
        if second
        else ""
    )
    turbulence_residuals = f" k 1e-4; {second} 1e-4;" if second else ""
    turbulence_relaxation = f" k 0.5; {second} 0.5;" if second else ""
    if layout.scalar_field:
        turbulence_solvers += (
            f"\n    {layout.scalar_field} {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }}"
        )
        turbulence_relaxation += f" {layout.scalar_field} 0.7;"
    if layout.pimple:
        (dest / "system" / "fvSolution").write_text(
            _foam_header("dictionary", "fvSolution")
            + f"""solvers
{{
    p {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.01; }}
    pFinal {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }}
    UFinal {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0; }}
}}
{TRANSIENT_PIMPLE_BLOCK}
"""
        )
    else:
        (dest / "system" / "fvSolution").write_text(
            _foam_header("dictionary", "fvSolution")
            + f"""solvers
{{
    p {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.01; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }}{turbulence_solvers}
}}
SIMPLE {{ nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; residualControl {{ p 1e-4; U 1e-4;{turbulence_residuals} }} }}
relaxationFactors {{ fields {{ p 0.3; }} equations {{ U 0.7;{turbulence_relaxation} }} }}
"""
        )

    should_validate = validate_mesh if validate_mesh is not None else mesh_gate_available()
    if should_validate:
        validate_seed_mesh(dest)

