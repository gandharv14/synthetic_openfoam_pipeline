from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

from synthetic_openfoam_pipeline.openfoam.runner import OpenFoamRunner, OpenFoamRunResult

MAX_EXCERPT_LINES = 40

# Lines worth surfacing as evidence: failures, mesh-quality findings, solver
# convergence behavior, and numerical-health warnings.
INTERESTING_LINE = re.compile(
    r"(FOAM FATAL|FOAM Warning|--> FOAM|Failed \d+ mesh check|mesh check|Mesh OK|"
    r"non-orthogonality|skewness|aspect ratio|continuity errors|Final residual|"
    r"not converged|diverg|\bnan\b|floating point|bounding|singularity|"
    r"cells:|faces:|points:)",
    re.IGNORECASE,
)

# Nondeterministic or banner noise that must never reach the instruction.
# (Keeps '--> FOAM ...' diagnostic lines; drops the ASCII-art header/footer.)
NOISE_LINE = re.compile(
    r"(ExecutionTime|ClockTime|^Date\b|^Host\b|^PID\b|^Exec\b|^Build\b|^Arch\b|"
    r"^nProcs\b|^Case\b|^fileModificationChecking|^allowSystemOperations|"
    r"^\s*\||^\s*/\*|^\s*\\\*|^\s*\*|^\s*={3,}|^\s*-{3,})"
)


def collect_symptom_evidence(case_dir: Path, timeout: int = 120) -> dict[str, Any]:
    """Run the broken starter case once and return a deterministic excerpt.

    Primary path: blockMesh/checkMesh/solver via OpenFoamRunner on a scratch
    copy, with the failing log (or the run's convergence/quality summary)
    distilled into a sanitized excerpt. When OpenFOAM is unavailable at
    generation time, fall back to a deterministic static audit of the case
    dictionaries so the instruction still carries concrete evidence.
    """
    if shutil.which("blockMesh") is None:
        return {"mode": "static-case-audit", "excerpt": static_case_audit(case_dir)}
    tmp = Path(tempfile.mkdtemp(prefix="synthetic-openfoam-evidence-"))
    try:
        scratch = tmp / "case"
        shutil.copytree(case_dir, scratch)
        result = OpenFoamRunner().run_case(scratch, timeout=timeout)
        excerpt = run_log_excerpt(result, scratch)
        if not excerpt.strip():
            return {"mode": "static-case-audit", "excerpt": static_case_audit(case_dir)}
        return {
            "mode": "openfoam-run",
            "excerpt": excerpt,
            "run_ok": result.ok,
            "failed_stage": result.failed_stage,
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_log_excerpt(result: OpenFoamRunResult, scratch: Path) -> str:
    logs = result.logs or {}
    if result.failed_stage and result.failed_stage in logs:
        header = f"[diagnostics: run failed during the '{result.failed_stage}' stage of blockMesh/checkMesh/solver]"
        selected = _interesting_lines(logs[result.failed_stage], scratch)
        if not selected:
            selected = _sanitize(logs[result.failed_stage], scratch)[-MAX_EXCERPT_LINES:]
    else:
        header = "[diagnostics: blockMesh/checkMesh/solver run summary]"
        selected = []
        for stage in ("checkMesh", "solve"):
            selected.extend(_interesting_lines(logs.get(stage, ""), scratch))
    if not selected:
        return ""
    return "\n".join([header, *selected[-MAX_EXCERPT_LINES:]])


def static_case_audit(case_dir: Path) -> str:
    lines = ["[case audit: key dictionary state of the starter case]"]
    control = _read(case_dir / "system" / "controlDict")
    for key in ("application", "endTime", "deltaT", "writeInterval"):
        value = _entry(control, key)
        if value is not None:
            lines.append(f"system/controlDict {key}: {value}")
    functions = _block_body(control, "functions")
    declared = ", ".join(re.findall(r"([A-Za-z_]\w*)\s*\{", functions)) if functions else ""
    lines.append(f"system/controlDict function objects: {declared or '(none declared)'}")
    div = re.search(r"div\(phi,U\)\s+([^;]+);", _read(case_dir / "system" / "fvSchemes"))
    if div:
        lines.append(f"system/fvSchemes div(phi,U): {div.group(1).strip()}")
    mesh = _read(case_dir / "system" / "blockMeshDict")
    for cells, grading in re.findall(r"\((\d+\s+\d+\s+\d+)\)\s+simpleGrading\s+\(([^)]+)\)", mesh):
        lines.append(f"system/blockMeshDict block: cells ({cells}) simpleGrading ({grading})")
    source = re.search(r"U\s+\(\([^)]+\)\s+0\);", _read(case_dir / "constant" / "fvOptions"))
    if source:
        lines.append(f"constant/fvOptions momentum source: {source.group(0)}")
    nu = _entry(_read(case_dir / "constant" / "transportProperties"), "nu")
    if nu is not None:
        lines.append(f"constant/transportProperties nu: {nu}")
    for field_name in ("U", "p", "T"):
        for patch, body in _patches(case_dir / "0" / field_name):
            lines.append(f"0/{field_name} boundary '{patch}': {body}")
    return "\n".join(lines)


def _sanitize(text: str, scratch: Path) -> list[str]:
    cleaned = text.replace(str(scratch), "<case>")
    lines: list[str] = []
    for raw in cleaned.splitlines():
        line = raw.rstrip()
        if not line or NOISE_LINE.search(line):
            continue
        lines.append(line)
    return lines


def _interesting_lines(text: str, scratch: Path) -> list[str]:
    return [line for line in _sanitize(text, scratch) if INTERESTING_LINE.search(line)]


def _read(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _entry(text: str, key: str) -> str | None:
    match = re.search(r"\b" + re.escape(key) + r"\s+([^;{}]+);", text)
    return match.group(1).strip() if match else None


def _block_body(text: str, name: str) -> str:
    match = re.search(r"\b" + re.escape(name) + r"\b\s*\{", text)
    if not match:
        return ""
    depth = 0
    for idx in range(match.end() - 1, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[match.end():idx]
    return ""


def _patches(path: Path) -> list[tuple[str, str]]:
    body = _block_body(_read(path), "boundaryField")
    if not body:
        return []
    patches: list[tuple[str, str]] = []
    idx = 0
    pattern = re.compile(r"([A-Za-z_][\w.]*)\s*\{")
    while idx < len(body):
        match = pattern.search(body, idx)
        if not match:
            break
        depth = 1
        end = match.end()
        while end < len(body) and depth:
            if body[end] == "{":
                depth += 1
            elif body[end] == "}":
                depth -= 1
            end += 1
        inner = body[match.end():end - 1]
        condensed = " ".join(" ".join(part.split()) for part in inner.strip().splitlines() if part.strip())
        patches.append((match.group(1), condensed))
        idx = end
    return patches
