from __future__ import annotations

from pathlib import Path


def replace_required(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise ValueError(f"Expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new))


def append_comment(path: Path, comment: str) -> None:
    path.write_text(path.read_text() + f"\n// {comment}\n")

