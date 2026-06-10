"""Rendering utilities for synthetic OpenFOAM review artifacts."""

from synthetic_openfoam_pipeline.rendering.problem_visualizer import (
    ProblemRender,
    TaskRenderSpec,
    discover_openfoam_tasks,
    render_all,
    render_problem,
    update_report,
)
from synthetic_openfoam_pipeline.rendering.physical_visualizer import (
    physical_render_metadata,
    render_all_physical,
    render_physical_problem,
)

__all__ = [
    "ProblemRender",
    "TaskRenderSpec",
    "discover_openfoam_tasks",
    "render_all",
    "render_all_physical",
    "render_physical_problem",
    "physical_render_metadata",
    "render_problem",
    "update_report",
]
