from __future__ import annotations

import zipfile
from pathlib import Path

from synthetic_openfoam_pipeline.config import PipelineConfig
from synthetic_openfoam_pipeline.schemas import TaskCandidate
from synthetic_openfoam_pipeline.utils.fs import ensure_dir


EXCLUDE_NAMES = {".DS_Store"}
EXCLUDE_PARTS = {"__MACOSX", "__pycache__", ".pytest_cache"}


class ZipExporter:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def export(self, candidate: TaskCandidate) -> Path:
        if candidate.task_dir is None:
            raise ValueError("Candidate must be exported to a task directory before zipping")
        ensure_dir(self.config.zips_root, allow_template_output=self.config.allow_template_output)
        zip_path = self.config.zips_root / f"{candidate.task_id}.zip"
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(candidate.task_dir.rglob("*")):
                if path.name in EXCLUDE_NAMES or any(part in EXCLUDE_PARTS for part in path.parts):
                    continue
                if path.is_dir():
                    continue
                rel = Path(candidate.task_id) / path.relative_to(candidate.task_dir)
                zf.write(path, rel.as_posix())
        candidate.zip_path = zip_path
        return zip_path

