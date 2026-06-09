from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from synthetic_openfoam_pipeline.schemas import SeedCaseSpec
from synthetic_openfoam_pipeline.utils.hashing import stable_hash


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
                "transient_setup_repair",
                "mesh_resolution_tradeoff",
            ],
            runtime_estimate_sec=30,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01},
        ),
        SeedCaseSpec(
            seed_id="cavity-incompressible",
            case_kind="cavity",
            solver="simpleFoam",
            compatible_families=["postprocessing_instrumentation", "mesh_quality_repair"],
            runtime_estimate_sec=25,
            parameters={"lid_velocity": 1.0, "viscosity": 0.01},
        ),
        SeedCaseSpec(
            seed_id="heated-scalar-channel",
            case_kind="source_channel",
            solver="simpleFoam",
            compatible_families=["source_term_calibration"],
            runtime_estimate_sec=35,
            parameters={"source_strength": 0.2, "viscosity": 0.01},
        ),
        SeedCaseSpec(
            seed_id="backward-facing-step",
            case_kind="backward_facing_step",
            solver="simpleFoam",
            compatible_families=[
                "boundary_condition_repair",
                "solver_convergence_repair",
                "mesh_quality_repair",
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
            ],
            runtime_estimate_sec=35,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01, "source_strength": 0.2},
        ),
        SeedCaseSpec(
            seed_id="pipe-bend",
            case_kind="pipe_bend",
            solver="simpleFoam",
            compatible_families=[
                "boundary_condition_repair",
                "solver_convergence_repair",
                "mesh_quality_repair",
            ],
            runtime_estimate_sec=35,
            parameters={"inlet_velocity": 1.0, "viscosity": 0.01, "bend_angle_degrees": 90},
        ),
    ]


def compatible_seed_specs(family: str) -> list[SeedCaseSpec]:
    return [spec for spec in seed_specs() if family in spec.compatible_families]


def pick_seed_for_family(family: str, selector: Any = None) -> SeedCaseSpec:
    specs = compatible_seed_specs(family)
    if not specs:
        specs = [seed_specs()[0]]
    if selector is None:
        return specs[0]
    if isinstance(selector, int):
        return specs[selector % len(specs)]
    index = int(stable_hash({"family": family, "selector": selector}, 16), 16) % len(specs)
    return specs[index]


def _foam_header(cls: str, obj: str) -> str:
    return f"FoamFile {{ version 2.0; format ascii; class {cls}; object {obj}; }}\n"


def _function_objects(*, probes: bool = True, field_min_max: bool = True, field_average: bool = False) -> str:
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
    return "functions\n{\n" + "\n".join(blocks) + "\n}\n"


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


def _default_channel_u_boundaries() -> str:
    return """    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0); }
    walls { type noSlip; }
    frontAndBack { type empty; }"""


def _default_channel_p_boundaries() -> str:
    return """    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls { type zeroGradient; }
    frontAndBack { type empty; }"""


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


def _case_layout(case_kind: str) -> CaseLayout:
    if case_kind in {"channel", "source_channel", "porous_pressure_drop_channel"}:
        length = 2.0
        mesh_boundaries = """    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    walls { type wall; faces ((0 1 5 4) (3 7 6 2)); }"""
        source_strength = 0.2 if case_kind in {"source_channel", "porous_pressure_drop_channel"} else None
        return CaseLayout(
            length=length,
            vertices=_rect_vertices(length),
            blocks="    hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading (1.5 1 1)",
            mesh_boundaries=mesh_boundaries,
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(field_average=case_kind == "source_channel"),
            source_strength=source_strength,
        )
    if case_kind in {"cavity", "heated_cavity"}:
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
            blocks="    hex (0 1 2 3 4 5 6 7) (10 10 1) simpleGrading (1.5 1 1)",
            mesh_boundaries=mesh_boundaries,
            u_boundaries=_default_cavity_u_boundaries(),
            p_boundaries=_default_cavity_p_boundaries(),
            functions=_function_objects(),
            extra_fields=extra_fields,
        )
    if case_kind == "backward_facing_step":
        xs = [0.0, 1.0, 3.0]
        ys = [0.0, 0.4, 1.0]
        vertices, v = _grid_vertex_map(xs, ys)
        return CaseLayout(
            length=3.0,
            vertices=vertices,
            blocks="\n".join(
                [
                    _block(v, 0, 1, 1, 2, "8 6 1"),
                    _block(v, 1, 2, 0, 1, "16 4 1"),
                    _block(v, 1, 2, 1, 2, "16 6 1"),
                ]
            ),
            mesh_boundaries=f"""    inlet {{ type patch; faces ({_face(v, ((0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 0)))}); }}
    outlet {{ type patch; faces ({_face(v, ((2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 0, 1)))} {_face(v, ((2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 1, 1)))}); }}
    walls {{ type wall; faces ({_face(v, ((1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)))} {_face(v, ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)))} {_face(v, ((0, 2, 0), (0, 2, 1), (1, 2, 1), (1, 2, 0)))} {_face(v, ((1, 2, 0), (1, 2, 1), (2, 2, 1), (2, 2, 0)))}); }}""",
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(),
        )
    if case_kind == "diffuser_channel_expansion":
        xs = [0.0, 1.0, 3.0]
        ys = [0.0, 0.2, 0.8, 1.0]
        vertices, v = _grid_vertex_map(xs, ys)
        return CaseLayout(
            length=3.0,
            vertices=vertices,
            blocks="\n".join(
                [
                    _block(v, 0, 1, 1, 2, "8 6 1"),
                    _block(v, 1, 2, 0, 1, "16 3 1"),
                    _block(v, 1, 2, 1, 2, "16 6 1"),
                    _block(v, 1, 2, 2, 3, "16 3 1"),
                ]
            ),
            mesh_boundaries=f"""    inlet {{ type patch; faces ({_face(v, ((0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 0)))}); }}
    outlet {{ type patch; faces ({_face(v, ((2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 0, 1)))} {_face(v, ((2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 1, 1)))} {_face(v, ((2, 2, 0), (2, 3, 0), (2, 3, 1), (2, 2, 1)))}); }}
    walls {{ type wall; faces ({_face(v, ((1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)))} {_face(v, ((1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)))} {_face(v, ((0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)))} {_face(v, ((0, 2, 0), (0, 2, 1), (1, 2, 1), (1, 2, 0)))} {_face(v, ((1, 2, 0), (1, 3, 0), (1, 3, 1), (1, 2, 1)))} {_face(v, ((1, 3, 0), (1, 3, 1), (2, 3, 1), (2, 3, 0)))}); }}""",
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(),
        )
    if case_kind == "obstacle_wake":
        xs = [0.0, 1.0, 1.2, 3.0]
        ys = [0.0, 0.4, 0.6, 1.0]
        vertices, v = _grid_vertex_map(xs, ys)
        block_specs = [
            (0, 1, 0, 1, "8 4 1"),
            (0, 1, 1, 2, "8 3 1"),
            (0, 1, 2, 3, "8 4 1"),
            (1, 2, 0, 1, "3 4 1"),
            (1, 2, 2, 3, "3 4 1"),
            (2, 3, 0, 1, "16 4 1"),
            (2, 3, 1, 2, "16 3 1"),
            (2, 3, 2, 3, "16 4 1"),
        ]
        return CaseLayout(
            length=3.0,
            vertices=vertices,
            blocks="\n".join(_block(v, *spec) for spec in block_specs),
            mesh_boundaries=f"""    inlet {{ type patch; faces ({_face(v, ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)))} {_face(v, ((0, 1, 0), (0, 1, 1), (0, 2, 1), (0, 2, 0)))} {_face(v, ((0, 2, 0), (0, 2, 1), (0, 3, 1), (0, 3, 0)))}); }}
    outlet {{ type patch; faces ({_face(v, ((3, 0, 0), (3, 1, 0), (3, 1, 1), (3, 0, 1)))} {_face(v, ((3, 1, 0), (3, 2, 0), (3, 2, 1), (3, 1, 1)))} {_face(v, ((3, 2, 0), (3, 3, 0), (3, 3, 1), (3, 2, 1)))}); }}
    walls {{ type wall; faces ({_face(v, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)))} {_face(v, ((1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)))} {_face(v, ((2, 0, 0), (3, 0, 0), (3, 0, 1), (2, 0, 1)))} {_face(v, ((0, 3, 0), (0, 3, 1), (1, 3, 1), (1, 3, 0)))} {_face(v, ((1, 3, 0), (1, 3, 1), (2, 3, 1), (2, 3, 0)))} {_face(v, ((2, 3, 0), (2, 3, 1), (3, 3, 1), (3, 3, 0)))} {_face(v, ((1, 1, 0), (1, 2, 0), (1, 2, 1), (1, 1, 1)))} {_face(v, ((2, 1, 0), (2, 1, 1), (2, 2, 1), (2, 2, 0)))} {_face(v, ((1, 1, 0), (1, 1, 1), (2, 1, 1), (2, 1, 0)))} {_face(v, ((1, 2, 0), (2, 2, 0), (2, 2, 1), (1, 2, 1)))}); }}""",
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(),
        )
    if case_kind == "pipe_bend":
        xs = [0.0, 0.6, 1.0]
        ys = [0.0, 0.4, 1.4]
        vertices, v = _grid_vertex_map(xs, ys)
        return CaseLayout(
            length=1.4,
            vertices=vertices,
            blocks="\n".join(
                [
                    _block(v, 0, 1, 0, 1, "8 5 1"),
                    _block(v, 1, 2, 0, 1, "5 5 1"),
                    _block(v, 1, 2, 1, 2, "5 12 1"),
                ]
            ),
            mesh_boundaries=f"""    inlet {{ type patch; faces ({_face(v, ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)))}); }}
    outlet {{ type patch; faces ({_face(v, ((1, 2, 0), (1, 2, 1), (2, 2, 1), (2, 2, 0)))}); }}
    walls {{ type wall; faces ({_face(v, ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)))} {_face(v, ((1, 0, 0), (2, 0, 0), (2, 0, 1), (1, 0, 1)))} {_face(v, ((2, 0, 0), (2, 1, 0), (2, 1, 1), (2, 0, 1)))} {_face(v, ((2, 1, 0), (2, 2, 0), (2, 2, 1), (2, 1, 1)))} {_face(v, ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)))} {_face(v, ((1, 1, 0), (1, 2, 0), (1, 2, 1), (1, 1, 1)))}); }}""",
            u_boundaries=_default_channel_u_boundaries(),
            p_boundaries=_default_channel_p_boundaries(),
            functions=_function_objects(),
        )
    raise ValueError(f"Unknown seed case kind: {case_kind}")


def write_seed_case(dest: Path, spec: SeedCaseSpec) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for sub in ("0", "constant", "system"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    layout = _case_layout(spec.case_kind)

    (dest / "0" / "U").write_text(
        _foam_header("volVectorField", "U")
        + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
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
        _foam_header("dictionary", "transportProperties") + "transportModel Newtonian;\nnu 0.01;\n"
    )
    (dest / "constant" / "turbulenceProperties").write_text(
        _foam_header("dictionary", "turbulenceProperties") + "simulationType laminar;\n"
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
defaultPatch {{ name frontAndBack; type empty; }}
mergePatchPairs ();
"""
    )
    (dest / "system" / "controlDict").write_text(
        _foam_header("dictionary", "controlDict")
        + f"""application {spec.solver};
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 100;
deltaT 1;
writeControl timeStep;
writeInterval 20;
runTimeModifiable false;
{layout.functions or _function_objects()}"""
    )
    (dest / "system" / "fvSchemes").write_text(
        _foam_header("dictionary", "fvSchemes")
        + """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linearUpwind grad(U); div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
"""
    )
    (dest / "system" / "fvSolution").write_text(
        _foam_header("dictionary", "fvSolution")
        + """solvers
{
    p { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-7; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }
}
SIMPLE { nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; residualControl { p 1e-4; U 1e-4; } }
relaxationFactors { fields { p 0.3; } equations { U 0.7; } }
"""
    )

