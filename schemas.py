from __future__ import annotations

from dataclasses import asdict, dataclass, field
try:
    from enum import StrEnum
except ImportError:  # Python 3.9 compatibility on the default macOS interpreter.
    from enum import Enum

    class StrEnum(str, Enum):
        pass
from pathlib import Path
from typing import Any


class TaskStatus(StrEnum):
    CANDIDATE = "candidate"
    NEEDS_OPENFOAM_VALIDATION = "needs_openfoam_validation"
    NEEDS_CLAUDE_VALIDATION = "needs_claude_validation"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class RunnerMode(StrEnum):
    REAL = "real"


@dataclass(frozen=True)
class SeedCaseSpec:
    seed_id: str
    case_kind: str
    solver: str
    compatible_families: list[str]
    runtime_estimate_sec: int
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HiddenVariant:
    name: str
    changes: dict[str, Any]
    expected_overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExpectedMetrics:
    residual_max: float = 1e-4
    mass_balance_max: float = 2e-3
    pressure_drop_min: float = 0.05
    pressure_drop_max: float = 5.0
    postprocessing_required: list[str] = field(default_factory=list)
    source_target: float | None = None
    tolerance: float = 0.05


@dataclass(frozen=True)
class ForbiddenEdit:
    path: str
    sha256: str
    description: str


@dataclass(frozen=True)
class DictionaryEditSpec:
    path: str
    old: str
    new: str
    description: str
    mode: str = "replace"


@dataclass(frozen=True)
class CorruptionSpec:
    steps: list[DictionaryEditSpec]
    description: str


@dataclass(frozen=True)
class OracleRepairSpec:
    steps: list[DictionaryEditSpec]
    description: str


@dataclass(frozen=True)
class ForbiddenEditSpec:
    path: str
    description: str = "Must preserve physical input"


@dataclass(frozen=True)
class OpenFoamTaskSpec:
    family: str
    seed_case: SeedCaseSpec
    corruption: CorruptionSpec
    oracle_repair: OracleRepairSpec
    hidden_variant_grid: list[HiddenVariant]
    expected_metrics: ExpectedMetrics
    forbidden_edits: list[ForbiddenEditSpec]


@dataclass
class TaskCandidate:
    task_id: str
    family: str
    seed: int
    seed_case: SeedCaseSpec
    skill_graph: dict[str, Any]
    hidden_variants: list[HiddenVariant]
    expected_metrics: ExpectedMetrics
    forbidden_edits: list[ForbiddenEdit]
    prompt: str
    rationale: str
    starter_case_dir: Path
    oracle_case_dir: Path
    status: TaskStatus = TaskStatus.CANDIDATE
    task_dir: Path | None = None
    zip_path: Path | None = None
    generation_report: dict[str, Any] = field(default_factory=dict)
    task_spec: OpenFoamTaskSpec | None = None

    def to_manifest(self) -> dict[str, Any]:
        data = {
            "task_id": self.task_id,
            "family": self.family,
            "seed": self.seed,
            "seed_case": asdict(self.seed_case),
            "skill_graph": self.skill_graph,
            "hidden_variants": [asdict(v) for v in self.hidden_variants],
            "expected_metrics": asdict(self.expected_metrics),
            "forbidden_edits": [asdict(e) for e in self.forbidden_edits],
            "rationale": self.rationale,
            "status": self.status.value,
            "task_dir": str(self.task_dir) if self.task_dir else None,
            "zip_path": str(self.zip_path) if self.zip_path else None,
        }
        if self.generation_report:
            data["generation_report"] = self.generation_report
        return data


@dataclass
class ScoreResult:
    score: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    task_id: str
    status: TaskStatus
    oracle: ScoreResult
    naive: ScoreResult
    hidden_passed: bool
    runner_mode: RunnerMode
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "oracle": asdict(self.oracle),
            "naive": asdict(self.naive),
            "hidden_passed": self.hidden_passed,
            "runner_mode": self.runner_mode.value,
            "issues": self.issues,
        }


@dataclass
class TargetAttemptResult:
    target: str
    attempt: int
    passed: bool
    available: bool = True
    status: str = "completed"
    score: float = 0.0
    transcript_path: str | None = None
    error: str | None = None


@dataclass
class AdversarialResult:
    task_id: str
    target: str
    status: TaskStatus
    attempts: list[TargetAttemptResult]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "target": self.target,
            "status": self.status.value,
            "attempts": [asdict(a) for a in self.attempts],
            "reason": self.reason,
        }

