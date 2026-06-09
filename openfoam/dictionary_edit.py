from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def replace_required(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if old not in text:
        raise ValueError(f"Expected text not found in {path}: {old!r}")
    path.write_text(text.replace(old, new))


def append_comment(path: Path, comment: str) -> None:
    path.write_text(path.read_text() + f"\n// {comment}\n")


def _num(value: float | int) -> str:
    return f"{float(value):g}"


def foam_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return "(" + " ".join(_num(v) for v in value) + ")"
    return _num(value)


def _block_span(text: str, name: str, *, start: int = 0) -> tuple[int, int, str] | None:
    match = re.search(rf"\b{re.escape(name)}\b\s*\{{", text[start:])
    if not match:
        return None
    block_start = start + match.start()
    pos = start + match.end() - 1
    depth = 0
    for idx in range(pos, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return block_start, idx + 1, text[pos + 1 : idx]
    raise ValueError(f"Unclosed OpenFOAM dictionary block: {name}")


def _entry_map(snippet: str) -> dict[str, str]:
    body_match = re.search(r"\{(.*)\}", snippet, flags=re.S)
    body = body_match.group(1) if body_match else snippet
    return {key: value.strip() for key, value in re.findall(r"\b([A-Za-z_][\w]*)\s+([^;{}]+);", body)}


def patch_entries(path: Path, patch: str) -> dict[str, str]:
    text = path.read_text()
    boundary = _block_span(text, "boundaryField")
    if boundary is None:
        return {}
    patch_block = _block_span(boundary[2], patch)
    if patch_block is None:
        return {}
    return _entry_map(patch_block[2])


def set_patch(path: Path, patch: str, entries: dict[str, str]) -> None:
    text = path.read_text()
    boundary = _block_span(text, "boundaryField")
    if boundary is None:
        raise ValueError(f"boundaryField block not found in {path}")
    patch_block = _block_span(boundary[2], patch)
    if patch_block is None:
        raise ValueError(f"Patch {patch!r} not found in {path}")
    order = [key for key in ("type", "inletValue", "value") if key in entries]
    order.extend(key for key in entries if key not in order)
    line = f"    {patch} {{ " + " ".join(f"{key} {entries[key]};" for key in order) + " }"
    body = boundary[2]
    new_body = body[: patch_block[0]] + line + body[patch_block[1] :]
    path.write_text(text[: boundary[0]] + f"boundaryField\n{{{new_body}}}" + text[boundary[1] :])


def set_internal_field(path: Path, value: Any) -> None:
    text = path.read_text()
    next_text, count = re.subn(r"\binternalField\s+uniform\s+[^;]+;", f"internalField uniform {foam_value(value)};", text, count=1)
    if count != 1:
        raise ValueError(f"internalField uniform entry not found in {path}")
    path.write_text(next_text)


def set_transport_nu(path: Path, value: float) -> None:
    _replace_entry(path, "nu", f"{value:.6f}")


def set_control_entry(path: Path, key: str, value: Any) -> None:
    _replace_entry(path, key, foam_value(value))


def control_entry(path: Path, key: str) -> str | None:
    match = re.search(rf"\b{re.escape(key)}\s+([^;]+);", path.read_text())
    return match.group(1).strip() if match else None


def _replace_entry(path: Path, key: str, value: str) -> None:
    text = path.read_text()
    next_text, count = re.subn(rf"(\b{re.escape(key)}\s+)([^;]+);", rf"\g<1>{value};", text, count=1)
    if count != 1:
        raise ValueError(f"Entry {key!r} not found in {path}")
    path.write_text(next_text)


def set_probe_location(path: Path, location: list[float] | tuple[float, float, float]) -> None:
    text = path.read_text()
    loc = " ".join(_num(v) for v in location)
    next_text, count = re.subn(r"\bprobeLocations\s*\(\([^)]+\)\);", f"probeLocations (({loc}));", text, count=1)
    if count != 1:
        raise ValueError(f"probeLocations entry not found in {path}")
    path.write_text(next_text)


def set_functions_block(path: Path, functions: str) -> None:
    text = path.read_text()
    block = _block_span(text, "functions")
    if block is None:
        path.write_text(text.rstrip() + "\n" + functions)
        return
    path.write_text(text[: block[0]] + functions.rstrip() + "\n" + text[block[1] :])


def set_div_phi_u_scheme(path: Path, scheme: str) -> None:
    text = path.read_text()
    next_text, count = re.subn(r"\bdiv\(phi,U\)\s+[^;]+;", f"div(phi,U) {scheme};", text, count=1)
    if count != 1:
        raise ValueError(f"div(phi,U) scheme not found in {path}")
    path.write_text(next_text)


def set_momentum_source(path: Path, value: float) -> None:
    text = path.read_text()
    next_text, count = re.subn(r"\bU\s+\(\([^)]+\)\s+0\);", f"U (({value:.3f} 0 0) 0);", text, count=1)
    if count != 1:
        raise ValueError(f"Momentum source U entry not found in {path}")
    path.write_text(next_text)


def momentum_source(path: Path) -> float | None:
    match = re.search(r"\bU\s+\(\(\s*([-+0-9.eE]+)\s+0\s+0\s*\)\s+0\);", path.read_text())
    return float(match.group(1)) if match else None


def set_all_simple_grading(path: Path, grading: Any) -> None:
    text = path.read_text()
    next_text, count = re.subn(r"\bsimpleGrading\s+\([^)]+\)", f"simpleGrading {foam_value(grading)}", text)
    if count < 1:
        raise ValueError(f"simpleGrading entry not found in {path}")
    path.write_text(next_text)


def first_simple_grading_x(path: Path) -> float | None:
    match = re.search(r"\bsimpleGrading\s+\(\s*([-+0-9.eE]+)", path.read_text())
    return float(match.group(1)) if match else None


def replace_cell_count(path: Path, old: tuple[int, int, int], new: tuple[int, int, int]) -> None:
    text = path.read_text()
    old_text = f"({old[0]} {old[1]} {old[2]}) simpleGrading"
    new_text = f"({new[0]} {new[1]} {new[2]}) simpleGrading"
    if old_text not in text:
        raise ValueError(f"Cell-count entry {old_text!r} not found in {path}")
    path.write_text(text.replace(old_text, new_text, 1))


def block_mesh_cells(path: Path) -> int:
    total = 0
    for nx, ny, nz in re.findall(r"\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s+simpleGrading", path.read_text()):
        total += int(nx) * int(ny) * int(nz)
    return total


def apply_structured_edit(path: Path, old: str, new: str, *, mode: str = "replace") -> bool:
    rel = path.as_posix()
    if mode == "replace_tail" and path.name == "controlDict":
        set_functions_block(path, new)
        return True
    if rel.endswith("/0/U"):
        if "internalField" in old:
            set_internal_field(path, _entry_map(new).get("internalField", new))
            return True
        if "inlet {" in old or "inlet {" in new:
            set_patch(path, "inlet", _entry_map(new))
            return True
    if rel.endswith("/0/p") and ("outlet {" in old or "outlet {" in new):
        set_patch(path, "outlet", _entry_map(new))
        return True
    if rel.endswith("/0/T") and ("movingWall {" in old or "movingWall {" in new):
        set_patch(path, "movingWall", _entry_map(new))
        return True
    if path.name == "transportProperties" and "nu" in old:
        values = re.findall(r"[-+0-9.eE]+", new)
        if values:
            set_transport_nu(path, float(values[-1]))
            return True
    if path.name == "fvOptions" and "U ((" in old:
        value = momentum_source_value_from_text(new)
        if value is not None:
            set_momentum_source(path, value)
            return True
    if path.name == "controlDict":
        if old.startswith("endTime"):
            set_control_entry(path, "endTime", new.split()[1].rstrip(";"))
            return True
        if old.startswith("writeInterval"):
            set_control_entry(path, "writeInterval", new.split()[1].rstrip(";"))
            return True
        if "functions" in old or "functions" in new:
            set_functions_block(path, new)
            return True
    if path.name == "fvSchemes" and "div(phi,U)" in old:
        scheme = new.split("div(phi,U)", 1)[1].strip().rstrip(";")
        set_div_phi_u_scheme(path, scheme)
        return True
    if path.name == "blockMeshDict":
        cell_old = _cell_tuple(old)
        cell_new = _cell_tuple(new)
        if cell_old and cell_new:
            replace_cell_count(path, cell_old, cell_new)
            return True
        grading = _grading_tuple(new)
        if grading:
            set_all_simple_grading(path, grading)
            return True
    return False


def momentum_source_value_from_text(text: str) -> float | None:
    match = re.search(r"\bU\s+\(\(\s*([-+0-9.eE]+)\s+0\s+0\s*\)\s+0\);", text)
    return float(match.group(1)) if match else None


def _cell_tuple(text: str) -> tuple[int, int, int] | None:
    match = re.search(r"\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s+simpleGrading", text)
    return (int(match.group(1)), int(match.group(2)), int(match.group(3))) if match else None


def _grading_tuple(text: str) -> tuple[float, ...] | None:
    match = re.search(r"\bsimpleGrading\s+\(([^)]+)\)", text)
    if not match:
        return None
    return tuple(float(value) for value in match.group(1).split())

