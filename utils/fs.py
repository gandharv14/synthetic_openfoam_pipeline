from __future__ import annotations

import json
import os
import shutil
import tarfile
from pathlib import Path
from typing import Any


TEMPLATE_DIRNAME = "lbx-rl-tasks-iso-template"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_within(child: Path, parent: Path) -> bool:
    try:
        return os.path.commonpath([child.resolve(), parent.resolve()]) == str(parent.resolve())
    except ValueError:
        return False


def ensure_not_in_template(path: Path, *, allow_template_output: bool | Path = False) -> None:
    resolved = path.resolve()
    if TEMPLATE_DIRNAME not in resolved.parts:
        return
    if allow_template_output is True:
        return
    if isinstance(allow_template_output, Path) and _is_within(resolved, allow_template_output):
        return
    raise ValueError(f"Refusing to write generated pipeline output inside {TEMPLATE_DIRNAME}: {path}")


def ensure_dir(path: Path, *, allow_template_output: bool | Path = False) -> Path:
    ensure_not_in_template(path, allow_template_output=allow_template_output)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Any, *, allow_template_output: bool | Path = False) -> None:
    ensure_not_in_template(path, allow_template_output=allow_template_output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def copytree_clean(src: Path, dst: Path, *, allow_template_output: bool | Path = False) -> None:
    ensure_not_in_template(dst, allow_template_output=allow_template_output)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def make_tar_gz(
    source_dir: Path,
    archive_path: Path,
    arcname: str = "fixed_case",
    *,
    allow_template_output: bool | Path = False,
) -> None:
    ensure_not_in_template(archive_path, allow_template_output=allow_template_output)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(source_dir.rglob("*")):
            if (
                path.name in {".DS_Store"}
                or ".synthetic_openfoam" in path.parts
                or "__MACOSX" in path.parts
                or "__pycache__" in path.parts
            ):
                continue
            tar.add(path, arcname=str(Path(arcname) / path.relative_to(source_dir)))


def safe_extract_tar(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as tar:
        root = output_dir.resolve()
        for member in tar.getmembers():
            target = (output_dir / member.name).resolve()
            if os.path.commonpath([root, target]) != str(root):
                raise ValueError(f"Unsafe path in archive: {member.name}")
        tar.extractall(output_dir)


def chmod_executable(path: Path) -> None:
    path.chmod(path.stat().st_mode | 0o111)

