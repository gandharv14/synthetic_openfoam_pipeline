"""Synthetic OpenFOAM RL task-generation pipeline."""

from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.schemas import TaskCandidate, TaskStatus

__all__ = ["PipelineConfig", "TaskCandidate", "TaskStatus"]
__version__ = "0.1.0"

