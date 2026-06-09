from __future__ import annotations

import shutil
from pathlib import Path

from synthetic_openfoam_pipeline.schemas import SeedCaseSpec


def seed_specs() -> list[SeedCaseSpec]:
    return [
        SeedCaseSpec(
            seed_id="laminar-channel",
            case_kind="channel",
            solver="simpleFoam",
            compatible_families=["boundary_condition_repair", "solver_convergence_repair"],
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
    ]


def pick_seed_for_family(family: str) -> SeedCaseSpec:
    for spec in seed_specs():
        if family in spec.compatible_families:
            return spec
    return seed_specs()[0]


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


def write_seed_case(dest: Path, spec: SeedCaseSpec) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    for sub in ("0", "constant", "system"):
        (dest / sub).mkdir(parents=True, exist_ok=True)

    is_channel = spec.case_kind in {"channel", "source_channel"}
    length = 2 if is_channel else 1
    grading = "1.5 1 1"
    if is_channel:
        u_boundaries = """    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type inletOutlet; inletValue uniform (0 0 0); value uniform (1 0 0); }
    walls { type noSlip; }
    frontAndBack { type empty; }"""
        p_boundaries = """    inlet { type zeroGradient; }
    outlet { type fixedValue; value uniform 0; }
    walls { type zeroGradient; }
    frontAndBack { type empty; }"""
        mesh_boundaries = """    inlet { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((1 2 6 5)); }
    walls { type wall; faces ((0 1 5 4) (3 7 6 2)); }"""
    else:
        u_boundaries = """    movingWall { type fixedValue; value uniform (1 0 0); }
    walls { type noSlip; }
    frontAndBack { type empty; }"""
        p_boundaries = """    movingWall { type zeroGradient; }
    walls { type zeroGradient; }
    frontAndBack { type empty; }"""
        mesh_boundaries = """    movingWall { type wall; faces ((3 7 6 2)); }
    walls { type wall; faces ((0 1 5 4) (0 4 7 3) (1 2 6 5)); }"""

    (dest / "0" / "U").write_text(
        _foam_header("volVectorField", "U")
        + f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (1 0 0);
boundaryField
{{
{u_boundaries}
}}
"""
    )
    (dest / "0" / "p").write_text(
        _foam_header("volScalarField", "p")
        + f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
{p_boundaries}
}}
"""
    )
    (dest / "constant" / "transportProperties").write_text(
        _foam_header("dictionary", "transportProperties") + "transportModel Newtonian;\nnu 0.01;\n"
    )
    (dest / "constant" / "turbulenceProperties").write_text(
        _foam_header("dictionary", "turbulenceProperties") + "simulationType laminar;\n"
    )
    source_strength = 0.2 if spec.case_kind == "source_channel" else None
    (dest / "constant" / "fvOptions").write_text(_fv_options(source_strength))
    (dest / "system" / "blockMeshDict").write_text(
        _foam_header("dictionary", "blockMeshDict")
        + f"""scale 1;
vertices ((0 0 0) ({length} 0 0) ({length} 1 0) (0 1 0) (0 0 0.1) ({length} 0 0.1) ({length} 1 0.1) (0 1 0.1));
blocks (hex (0 1 2 3 4 5 6 7) (20 10 1) simpleGrading ({grading}));
edges ();
boundary
(
{mesh_boundaries}
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
{_function_objects(field_average=spec.case_kind == "source_channel")}"""
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

