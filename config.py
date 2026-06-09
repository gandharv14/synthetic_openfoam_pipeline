from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_FAMILIES = [
    "boundary_condition_repair",
    "postprocessing_instrumentation",
    "solver_convergence_repair",
    "mesh_quality_repair",
    "source_term_calibration",
    "flow_rate_targeting",
    "pressure_drop_calibration",
    "transient_setup_repair",
    "force_coeff_instrumentation",
    "scalar_transport_repair",
    "mesh_resolution_tradeoff",
]


@dataclass(frozen=True)
class PipelineConfig:
    count: int = 5
    seed: int = 123
    output_root: Path = Path("generated_openfoam_tasks")
    template_root: Path = Path("lbx-rl-tasks-iso-template")
    problems_root_override: Path | None = None
    mirror_problems: bool = True
    mirror_problems_root_override: Path | None = None
    use_template_problems: bool = True
    families: list[str] = field(default_factory=lambda: list(DEFAULT_FAMILIES))
    naive_threshold: float = 0.45
    oracle_threshold: float = 0.999
    attempts: int = 3
    harness_timeout_sec: int = 300
    zip_tasks: bool = False
    target: str = "claude-harness"
    allow_template_output: bool = False
    skip_harness_ground_truth: bool = False
    prompt_rewrite_enabled: bool = False
    prompt_rewrite_model: str | None = None

    @property
    def problems_root(self) -> Path:
        if self.problems_root_override is not None:
            return self.problems_root_override
        if self.use_template_problems:
            return self.template_root / "problems"
        return self.output_root / "problems"

    @property
    def zips_root(self) -> Path:
        return self.output_root / "zips"

    @property
    def reports_root(self) -> Path:
        return self.output_root / "reports"

    @property
    def template_problems_root(self) -> Path:
        return self.template_root / "problems"

    @property
    def mirror_problems_root(self) -> Path | None:
        if not self.mirror_problems:
            return None
        if self.mirror_problems_root_override is not None:
            return self.mirror_problems_root_override
        return self.template_root.parent / "synthetic_openfoam_pipeline" / "problems"

    @property
    def template_write_root(self) -> bool | Path:
        if self.allow_template_output or self.use_template_problems:
            return self.template_problems_root
        return False


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [part.strip().strip("\"'") for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.strip("\"'")


def _load_tiny_yaml(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = _parse_scalar(value)
    return data


def load_config(path: str | Path | None = None, **overrides: Any) -> PipelineConfig:
    data: dict[str, Any] = {}
    if path:
        config_path = Path(path)
        if config_path.suffix.lower() == ".json":
            data.update(json.loads(config_path.read_text()))
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            data.update(_load_tiny_yaml(config_path))
        else:
            raise ValueError(f"Unsupported config extension: {config_path.suffix}")
    data.update({k: v for k, v in overrides.items() if v is not None})
    if "output_root" in data:
        data["output_root"] = Path(data["output_root"])
    if "template_root" in data:
        data["template_root"] = Path(data["template_root"])
    if "problems_root_override" in data:
        data["problems_root_override"] = Path(data["problems_root_override"])
    if "problems_root" in data:
        data["problems_root_override"] = Path(data.pop("problems_root"))
    if "mirror_problems_root_override" in data and data["mirror_problems_root_override"] is not None:
        data["mirror_problems_root_override"] = Path(data["mirror_problems_root_override"])
    if "mirror_problems_root" in data:
        mirror_problems_root = data.pop("mirror_problems_root")
        if mirror_problems_root is not None:
            data["mirror_problems_root_override"] = Path(mirror_problems_root)
    if isinstance(data.get("families"), str):
        data["families"] = [part.strip() for part in data["families"].split(",") if part.strip()]
    return PipelineConfig(**data)

