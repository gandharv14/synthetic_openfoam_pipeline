from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from synthetic_openfoam_pipeline.rendering.problem_visualizer import (
    ProblemRender,
    TaskRenderSpec,
    discover_openfoam_tasks,
    encode_frames,
    frame_fracs,
    graded_positions,
    load_task_spec,
    plotting_modules,
    save_frame,
)

PHYSICAL_MESH_FAMILIES = {"mesh_quality_repair", "mesh_resolution_tradeoff"}

PHYSICAL_CAPTIONS = {
    "flow_rate_targeting": "Physical OpenFOAM render of the repaired channel case, showing the solved flow response used to verify the inlet flow target.",
    "pressure_drop_calibration": "Physical OpenFOAM render of the repaired channel case, emphasizing the pressure field after restoring the outlet datum.",
    "transient_setup_repair": "Physical OpenFOAM render of the repaired case output cadence, using solver-written fields when available.",
    "force_coeff_instrumentation": "Physical OpenFOAM render of the repaired flow context behind the force-coefficient review evidence.",
    "scalar_transport_repair": "Physical OpenFOAM render of the repaired scalar boundary condition and resulting temperature field.",
    "mesh_quality_repair": "Physical mesh render comparing the broken starter blockMesh layout with the oracle repaired mesh.",
    "mesh_resolution_tradeoff": "Physical mesh render comparing the over-refined starter grid with the compact oracle repaired grid.",
}


def render_all_physical(
    tasks_root: Path,
    assets_dir: Path,
    *,
    task_ids: Iterable[str] | None = None,
    frames: int = 36,
    fps: int = 12,
    runner: str = "auto",
    timeout_sec: int = 900,
    fallback_to_mesh: bool = True,
    render_end_time: float = 250.0,
    render_write_interval: float = 5.0,
) -> list[ProblemRender]:
    selected = set(task_ids or [])
    renders = []
    for spec in discover_openfoam_tasks(tasks_root):
        if selected and spec.task_id not in selected:
            continue
        renders.append(
            render_physical_problem(
                spec.task_dir,
                assets_dir,
                frames=frames,
                fps=fps,
                runner=runner,
                timeout_sec=timeout_sec,
                fallback_to_mesh=fallback_to_mesh,
                render_end_time=render_end_time,
                render_write_interval=render_write_interval,
            )
        )
    return renders


def render_physical_problem(
    task_dir: Path,
    assets_dir: Path,
    *,
    frames: int = 36,
    fps: int = 12,
    runner: str = "auto",
    timeout_sec: int = 900,
    fallback_to_mesh: bool = True,
    render_end_time: float = 250.0,
    render_write_interval: float = 5.0,
) -> ProblemRender:
    spec = load_task_spec(task_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)
    render = physical_render_metadata(spec, assets_dir)

    with tempfile.TemporaryDirectory(prefix=f"{spec.task_id}-physical-") as tmp_name:
        tmp = Path(tmp_name)
        fixed_archive = build_oracle_case_archive(spec.task_dir, tmp, timeout_sec)
        try:
            if runner == "mesh" or spec.family in PHYSICAL_MESH_FAMILIES:
                render_mesh_video(spec, spec.task_dir / "data" / "starter_case.tar.gz", fixed_archive, render.path, frames, fps, tmp)
            else:
                render_solved_video(
                    spec,
                    fixed_archive,
                    render.path,
                    frames,
                    fps,
                    runner,
                    timeout_sec,
                    tmp,
                    render_end_time,
                    render_write_interval,
                )
        except Exception as exc:
            if not fallback_to_mesh:
                raise
            render_mesh_video(spec, spec.task_dir / "data" / "starter_case.tar.gz", fixed_archive, render.path, frames, fps, tmp)
            render = replace(
                render,
                caption=render.caption + f" Solver-field rendering was unavailable, so this MP4 falls back to the actual repaired mesh ({type(exc).__name__}).",
            )
    return render


def physical_render_metadata(spec: TaskRenderSpec, assets_dir: Path) -> ProblemRender:
    return ProblemRender(
        task_id=spec.task_id,
        family=spec.family,
        title=spec.title,
        path=Path(assets_dir) / f"{spec.task_id}-physical.mp4",
        caption=PHYSICAL_CAPTIONS.get(
            spec.family,
            "Physical OpenFOAM render generated from the repaired case dictionaries and solver outputs when available.",
        ),
        marker="physical-render",
        label="Physical OpenFOAM render",
        aria_label_suffix="physical OpenFOAM render",
    )


def build_oracle_case_archive(task_dir: Path, tmp: Path, timeout_sec: int) -> Path:
    solution = task_dir / "solution" / "solve.sh"
    starter = task_dir / "data" / "starter_case.tar.gz"
    if not solution.exists():
        return starter

    out = tmp / "oracle_out"
    out.mkdir(parents=True, exist_ok=True)
    env = {
        **os.environ,
        "LBT_DATA_DIR": str((task_dir / "data").resolve()),
        "LBT_OUTPUT_DIR": str(out.resolve()),
    }
    result = subprocess.run(
        ["bash", str(solution.resolve())],
        text=True,
        capture_output=True,
        check=False,
        timeout=min(timeout_sec, 300),
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"oracle repair failed for {task_dir.name}: {result.stderr[-800:]}")
    fixed = out / "fixed_case.tar.gz"
    if not fixed.exists():
        raise RuntimeError(f"oracle repair did not write {fixed}")
    return fixed


def render_solved_video(
    spec: TaskRenderSpec,
    fixed_archive: Path,
    output: Path,
    frames: int,
    fps: int,
    runner: str,
    timeout_sec: int,
    tmp: Path,
    render_end_time: float,
    render_write_interval: float,
) -> None:
    worker = tmp / "physical_render_worker.py"
    worker.write_text(PHYSICAL_WORKER_SCRIPT)
    actual_runner = choose_runner(runner, spec.task_dir)
    if actual_runner == "docker":
        run_worker_docker(spec, fixed_archive, output, worker, frames, fps, timeout_sec, tmp, render_end_time, render_write_interval)
    else:
        run_worker_local(spec, fixed_archive, output, worker, frames, fps, timeout_sec, render_end_time, render_write_interval)


def choose_runner(runner: str, task_dir: Path) -> str:
    if runner != "auto":
        return runner
    if shutil.which("blockMesh") or Path("/etc/solver-envs.d/openfoam.sh").exists() or Path("/usr/share/openfoam/etc/bashrc").exists():
        return "local"
    if shutil.which("docker") and template_task_dir(task_dir) is not None:
        return "docker"
    return "local"


def run_worker_local(
    spec: TaskRenderSpec,
    fixed_archive: Path,
    output: Path,
    worker: Path,
    frames: int,
    fps: int,
    timeout_sec: int,
    render_end_time: float,
    render_write_interval: float,
) -> None:
    cmd = [
        shutil.which("python3") or sys.executable,
        str(worker),
        "--case-archive",
        str(fixed_archive),
        "--family",
        spec.family,
        "--task-id",
        spec.task_id,
        "--title",
        spec.title,
        "--output",
        str(output),
        "--frames",
        str(frames),
        "--fps",
        str(fps),
        "--timeout",
        str(timeout_sec),
        "--render-end-time",
        f"{render_end_time:g}",
        "--render-write-interval",
        f"{render_write_interval:g}",
    ]
    result = subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout_sec + 120)
    if result.returncode != 0:
        raise RuntimeError(f"physical OpenFOAM render failed for {spec.task_id}: {result.stderr[-1200:]}")


def run_worker_docker(
    spec: TaskRenderSpec,
    fixed_archive: Path,
    output: Path,
    worker: Path,
    frames: int,
    fps: int,
    timeout_sec: int,
    tmp: Path,
    render_end_time: float,
    render_write_interval: float,
) -> None:
    template_task = template_task_dir(spec.task_dir)
    if template_task is None:
        raise RuntimeError(f"could not resolve template task directory for {spec.task_id}")
    template_root = template_task.parents[1]
    tag = existing_task_image(spec.task_id) or f"synthetic-openfoam-physical-{spec.task_id}:latest"
    platform = docker_platform_for_task(template_task)
    if tag.startswith("synthetic-openfoam-physical-"):
        build_cmd = [
            "docker",
            "build",
            "--pull=false",
            "-f",
            str((template_task / "environment" / "Dockerfile").resolve()),
            "--build-arg",
            f"PROBLEM_DIR=problems/{spec.task_id}",
            "-t",
            tag,
            ".",
        ]
        if platform:
            build_cmd[2:2] = ["--platform", platform]
        build = subprocess.run(
            build_cmd,
            cwd=template_root,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_sec,
        )
        if build.returncode != 0:
            raise RuntimeError(f"Docker build failed for {spec.task_id}: {build.stderr[-1200:]}")

    docker_archive = tmp / "fixed_case.tar.gz"
    docker_output = tmp / "physical.mp4"
    shutil.copy2(fixed_archive, docker_archive)
    run_cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            f"{tmp.resolve()}:/render_work",
            tag,
            "python3",
            "/render_work/physical_render_worker.py",
            "--case-archive",
            "/render_work/fixed_case.tar.gz",
            "--family",
            spec.family,
            "--task-id",
            spec.task_id,
            "--title",
            spec.title,
            "--output",
            "/render_work/physical.mp4",
            "--frames",
            str(frames),
            "--fps",
            str(fps),
            "--timeout",
            str(timeout_sec),
            "--render-end-time",
            f"{render_end_time:g}",
            "--render-write-interval",
            f"{render_write_interval:g}",
    ]
    if platform:
        run_cmd[2:2] = ["--platform", platform]
    run = subprocess.run(
        run_cmd,
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout_sec + 120,
    )
    if run.returncode != 0:
        raise RuntimeError(f"Docker physical render failed for {spec.task_id}: {run.stderr[-1200:]}")
    shutil.copy2(docker_output, output)


def existing_task_image(task_id: str) -> str | None:
    repo = f"lbx-rl-harness-{task_id}"
    result = subprocess.run(
        ["docker", "image", "ls", repo, "--format", "{{.Repository}}:{{.Tag}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    for line in result.stdout.splitlines():
        image = line.strip()
        if image and not image.endswith(":<none>"):
            return image
    return None


def docker_platform_for_task(task_dir: Path) -> str | None:
    dockerfile = task_dir / "environment" / "Dockerfile"
    image = dockerfile_base_image(dockerfile)
    if image is None:
        return None
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Os}}/{{.Architecture}}"],
        text=True,
        capture_output=True,
        check=False,
    )
    platform = result.stdout.strip()
    return platform if result.returncode == 0 and "/" in platform else None


def dockerfile_base_image(dockerfile: Path) -> str | None:
    if not dockerfile.exists():
        return None
    text = dockerfile.read_text()
    image_match = re.search(r"^\s*ARG\s+BASE_IMAGE=([^\s]+)", text, re.M)
    tag_match = re.search(r"^\s*ARG\s+BASE_TAG=([^\s]+)", text, re.M)
    if not image_match or not tag_match:
        return None
    return f"{image_match.group(1)}:{tag_match.group(1)}"


def template_task_dir(task_dir: Path) -> Path | None:
    task_dir = Path(task_dir).resolve()
    if task_dir.parent.name == "problems" and task_dir.parent.parent.name == "lbx-rl-tasks-iso-template":
        return task_dir
    root = task_dir
    for parent in [task_dir, *task_dir.parents]:
        candidate = parent / "lbx-rl-tasks-iso-template" / "problems" / task_dir.name
        if candidate.exists():
            return candidate
        root = parent
    sibling = root / "lbx-rl-tasks-iso-template" / "problems" / task_dir.name
    return sibling if sibling.exists() else None


def render_mesh_video(
    spec: TaskRenderSpec,
    starter_archive: Path,
    fixed_archive: Path,
    output: Path,
    frames: int,
    fps: int,
    tmp: Path,
) -> None:
    starter_case = extract_archive(starter_archive, tmp / "starter_case")
    fixed_case = extract_archive(fixed_archive, tmp / "fixed_case")
    starter_mesh = parse_block_mesh(starter_case / "system" / "blockMeshDict")
    fixed_mesh = parse_block_mesh(fixed_case / "system" / "blockMeshDict")

    frames_dir = tmp / "mesh_frames"
    frames_dir.mkdir(exist_ok=True)
    np, plt = plotting_modules()
    for index, frac in enumerate(frame_fracs(frames)):
        fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
        draw_mesh_panel(np, plt, axes[0], starter_mesh, "Starter mesh", frac, "#bc2f2f")
        draw_mesh_panel(np, plt, axes[1], fixed_mesh, "Oracle repaired mesh", frac, "#0f8a63")
        fig.suptitle(f"{spec.title} - physical blockMesh render ({spec.seed_case.get('seed_id', 'synthetic case')})", fontsize=15, weight="bold")
        save_frame(fig, frames_dir, index)
    encode_frames(frames_dir, output, fps)


def extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    children = [path for path in dest.iterdir() if path.is_dir()]
    return children[0] if children else dest


def parse_block_mesh(path: Path) -> dict:
    text = path.read_text()
    vertices_match = re.search(r"vertices\s*\((.*?)\);\s*blocks", text, re.S)
    block_match = re.search(
        r"hex\s*\(([^)]*)\)\s*\((\d+)\s+(\d+)\s+(\d+)\)\s+simpleGrading\s*\(([^)]*)\)",
        text,
        re.S,
    )
    if not vertices_match or not block_match:
        raise ValueError(f"could not parse blockMeshDict: {path}")
    vertices = [
        tuple(float(value) for value in match)
        for match in re.findall(r"\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\)", vertices_match.group(1))
    ]
    indices = [int(value) for value in block_match.group(1).split()]
    points = [vertices[index] for index in indices if index < len(vertices)]
    nx, ny, nz = (int(block_match.group(i)) for i in (2, 3, 4))
    grading_values = [float(value) for value in re.findall(r"[-+0-9.eE]+", block_match.group(5))]
    grading = tuple((grading_values + [1.0, 1.0, 1.0])[:3])
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return {
        "bounds": (min(xs), max(xs), min(ys), max(ys)),
        "cells": (nx, ny, nz),
        "grading": grading,
    }


def draw_mesh_panel(np, plt, ax, mesh: dict, title: str, frac: float, color: str) -> None:
    xmin, xmax, ymin, ymax = mesh["bounds"]
    nx, ny, nz = mesh["cells"]
    gx, gy, _ = mesh["grading"]
    x = xmin + (xmax - xmin) * graded_positions(np, nx, max(gx, 1e-6))
    y = ymin + (ymax - ymin) * graded_positions(np, ny, max(gy, 1e-6))
    for value in x:
        ax.plot([value, value], [ymin, ymax], color=color, alpha=0.35, lw=0.8)
    for value in y:
        ax.plot([xmin, xmax], [value, value], color=color, alpha=0.35, lw=0.8)
    sweep = xmin + (xmax - xmin) * frac
    ax.axvline(sweep, color="#2f6fed", lw=2.0, alpha=0.85)
    ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor="#152033", lw=1.4))
    ax.set_title(title, fontsize=12)
    ax.set_aspect("equal")
    ax.set_xlim(xmin - 0.05 * max(xmax - xmin, 1e-6), xmax + 0.05 * max(xmax - xmin, 1e-6))
    ax.set_ylim(ymin - 0.08 * max(ymax - ymin, 1e-6), ymax + 0.08 * max(ymax - ymin, 1e-6))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.text(
        0.03,
        0.96,
        f"{nx} x {ny} x {nz} cells\nsimpleGrading ({mesh['grading'][0]:g}, {mesh['grading'][1]:g}, {mesh['grading'][2]:g})",
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d8e0ec", "alpha": 0.92},
    )


PHYSICAL_WORKER_SCRIPT = r'''from __future__ import annotations

import argparse
import math
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

NUM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(dest)
    children = [path for path in dest.iterdir() if path.is_dir()]
    return children[0] if children else dest


def openfoam_env() -> str:
    return (
        "if [ -f /etc/solver-envs.d/openfoam.sh ]; then source /etc/solver-envs.d/openfoam.sh; "
        "elif [ -f /usr/share/openfoam/etc/bashrc ]; then source /usr/share/openfoam/etc/bashrc; "
        "elif ls /opt/openfoam*/etc/bashrc >/dev/null 2>&1; then source /opt/openfoam*/etc/bashrc; "
        "fi"
    )


def run_command(command: str, timeout: int) -> str:
    result = subprocess.run(
        ["bash", "-lc", f"{openfoam_env()} >/dev/null 2>&1 && {command}"],
        text=True,
        capture_output=True,
        check=False,
        timeout=timeout,
    )
    log = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {command}\n{log[-1200:]}")
    return log


def application(case: Path) -> str:
    match = re.search(r"\bapplication\s+([A-Za-z0-9_]+)\s*;", read_text(case / "system" / "controlDict"))
    return match.group(1) if match else "simpleFoam"


def strip_functions_for_solver(case: Path) -> str:
    control = case / "system" / "controlDict"
    original = read_text(control)
    if "functions" in original:
        control.write_text(re.sub(r"functions\s*\{.*\}\s*$", "functions {}\n", original, count=1, flags=re.S))
    return original


def set_entry(path: Path, key: str, value: str) -> None:
    text = read_text(path)
    next_text, count = re.subn(r"\b" + re.escape(key) + r"\s+[^;]+;", f"{key} {value};", text, count=1)
    if count == 0:
        next_text = text.rstrip() + f"\n{key} {value};\n"
    path.write_text(next_text)


def control_value(case: Path, key: str, default: float) -> float:
    match = re.search(r"\b" + re.escape(key) + r"\s+([-+0-9.eE]+)\s*;", read_text(case / "system" / "controlDict"))
    return float(match.group(1)) if match else default


def prepare_control_for_render(case: Path, render_end_time: float, render_write_interval: float) -> None:
    control = case / "system" / "controlDict"
    end_time = max(control_value(case, "endTime", render_end_time), render_end_time)
    set_entry(control, "endTime", f"{end_time:g}")
    set_entry(control, "writeControl", "timeStep")
    set_entry(control, "writeInterval", f"{max(render_write_interval, 1):g}")


def read_vector_internal(path: Path) -> np.ndarray:
    text = read_text(path)
    match = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\d*\s*\((.*?)\)\s*;", text, re.S)
    if match:
        values = [float(value) for value in NUM.findall(match.group(1))]
        return np.asarray(values, dtype=float).reshape(-1, 3)
    uniform = re.search(r"internalField\s+uniform\s*\(([^)]*)\)", text)
    if uniform:
        return np.asarray([[float(value) for value in NUM.findall(uniform.group(1))]], dtype=float)
    raise ValueError(f"could not parse vector field {path}")


def read_scalar_internal(path: Path) -> np.ndarray:
    text = read_text(path)
    match = re.search(r"internalField\s+nonuniform\s+List<scalar>\s*\d*\s*\((.*?)\)\s*;", text, re.S)
    if match:
        return np.asarray([float(value) for value in NUM.findall(match.group(1))], dtype=float)
    uniform = re.search(r"internalField\s+uniform\s+([-+0-9.eE]+)", text)
    if uniform:
        return np.asarray([float(uniform.group(1))], dtype=float)
    raise ValueError(f"could not parse scalar field {path}")


def field_for_family(family: str) -> tuple[str, str, str]:
    if family == "scalar_transport_repair":
        return "T", "scalar", "temperature"
    if family == "pressure_drop_calibration":
        return "p", "scalar", "pressure"
    return "U", "vector", "speed"


def time_dirs(case: Path, field: str) -> list[tuple[float, Path]]:
    times = []
    for path in case.iterdir():
        if not path.is_dir() or not (path / field).exists():
            continue
        try:
            value = float(path.name)
        except ValueError:
            continue
        times.append((value, path))
    return sorted(times) or [(0.0, case / "0")]


def field_values(path: Path, mode: str, count: int) -> tuple[np.ndarray, np.ndarray | None]:
    if mode == "vector":
        vectors = read_vector_internal(path)
        if len(vectors) == 1 and count > 1:
            vectors = np.repeat(vectors, count, axis=0)
        speed = np.linalg.norm(vectors[:, :2], axis=1)
        return speed, vectors
    values = read_scalar_internal(path)
    if len(values) == 1 and count > 1:
        values = np.repeat(values, count)
    return values, None


def residual_summary(log: str) -> str:
    values = [float(value) for value in re.findall(r"Final residual =\s*([-+0-9.eE]+)", log)]
    if not values:
        return "residuals unavailable"
    return f"max final residual {max(values):.2e}"


def encode(frames_dir: Path, output: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg") or "ffmpeg"
    result = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-vf",
            "scale=1280:720:flags=bicubic,format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "20",
            str(output),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr[-800:]}")


def render_case(args: argparse.Namespace) -> None:
    with tempfile.TemporaryDirectory(prefix="physical-openfoam-render-") as tmp_name:
        tmp = Path(tmp_name)
        case = extract_archive(Path(args.case_archive), tmp / "case")
        quoted_case = shlex.quote(str(case))
        run_command(f"blockMesh -case {quoted_case}", args.timeout)
        run_command(f"checkMesh -case {quoted_case}", args.timeout)
        prepare_control_for_render(case, args.render_end_time, args.render_write_interval)
        original_control = strip_functions_for_solver(case)
        solver_log = run_command(f"{application(case)} -case {quoted_case}", args.timeout)
        if original_control:
            (case / "system" / "controlDict").write_text(original_control)
        run_command(f"postProcess -case {quoted_case} -func writeCellCentres -time 0", args.timeout)

        centres = read_vector_internal(case / "0" / "C")
        field, mode, label = field_for_family(args.family)
        dirs = time_dirs(case, field)
        sample_positions = np.linspace(0, len(dirs) - 1, args.frames).astype(int)
        tri = mtri.Triangulation(centres[:, 0], centres[:, 1])
        frames_dir = tmp / "frames"
        frames_dir.mkdir()
        summary = residual_summary(solver_log)

        all_values = []
        for _, directory in dirs:
            values, _ = field_values(directory / field, mode, len(centres))
            all_values.append(values)
        vmin = min(float(np.nanmin(values)) for values in all_values)
        vmax = max(float(np.nanmax(values)) for values in all_values)
        if math.isclose(vmin, vmax):
            pad = max(abs(vmin) * 0.05, 1e-6)
            vmin -= pad
            vmax += pad
        levels = np.linspace(vmin, vmax, 32)

        for frame_index, sample_index in enumerate(sample_positions):
            time_value, directory = dirs[int(sample_index)]
            values, vectors = field_values(directory / field, mode, len(centres))
            fig, ax = plt.subplots(figsize=(12.8, 7.2), dpi=100, constrained_layout=True)
            contour = ax.tricontourf(tri, values, levels=levels, cmap="viridis", extend="both")
            if vectors is not None and len(vectors) == len(centres):
                stride = max(len(centres) // 60, 1)
                ax.quiver(
                    centres[::stride, 0],
                    centres[::stride, 1],
                    vectors[::stride, 0],
                    vectors[::stride, 1],
                    color="white",
                    alpha=0.8,
                    scale=35,
                    width=0.003,
                )
            fig.colorbar(contour, ax=ax, fraction=0.035, pad=0.02, label=label)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{args.title} - {label} field at t={time_value:g}", fontsize=13)
            ax.text(
                0.02,
                0.96,
                summary,
                transform=ax.transAxes,
                color="#152033",
                fontsize=10,
                va="top",
                bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#d8e0ec", "alpha": 0.92},
            )
            fig.savefig(frames_dir / f"frame_{frame_index:04d}.png", dpi=100)
            plt.close(fig)
        encode(frames_dir, Path(args.output), args.fps)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-archive", required=True)
    parser.add_argument("--family", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--render-end-time", type=float, default=250.0)
    parser.add_argument("--render-write-interval", type=float, default=5.0)
    render_case(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''
