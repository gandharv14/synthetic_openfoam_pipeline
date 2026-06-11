from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from openfoam_grader_lib import (
    apply_hidden_variant,
    family_repair_ok,
    forbidden_unchanged,
    load_json,
    required_files_present,
    select_runner,
    submission_case,
)


def _criterion(results: list[dict[str, Any]], cid: str, weight: float, passed: bool, detail: Any = None) -> float:
    value = weight if passed else 0.0
    results.append({"id": cid, "weight": weight, "passed": bool(passed), "score": value, "detail": detail})
    return value


def _result(score: float, criteria: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    detailed_metadata = dict(metadata)
    detailed_metadata["criteria"] = criteria
    return {
        "score": round(score, 6),
        "subscores": {item["id"]: item["score"] for item in criteria},
        "weights": {item["id"]: item["weight"] for item in criteria},
        "metadata": detailed_metadata,
    }


def compute_score(workspace: Path, trajectory: list[dict[str, Any]] | None, private: Path) -> dict[str, Any]:
    _ = trajectory
    workspace = Path(workspace)
    private = Path(private)
    criteria: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {"runner": "real-openfoam"}

    hidden_payload = load_json(private / "hidden_variants.json", []) or []
    # Per-task derived grids ship as {"axes", "derived_grid", "variants"};
    # legacy payloads are a bare variant list.
    hidden_variants = hidden_payload.get("variants") or [] if isinstance(hidden_payload, dict) else hidden_payload
    expected = load_json(private / "expected_metrics.json", {}) or {}
    metric_anchors = load_json(private / "metric_anchors.json", {}) or {}
    forbidden = load_json(private / "forbidden_edits.json", []) or []
    family = str(expected.get("family", ""))

    case_dir, submission_reason = submission_case(workspace)
    score = _criterion(criteria, "submission_exists_and_extracts", 0.10, case_dir is not None, submission_reason)
    if case_dir is None:
        return _result(score, criteria, metadata)

    required_ok, missing = required_files_present(case_dir)
    score += _criterion(criteria, "required_openfoam_files", 0.10, required_ok, missing)
    forbidden_ok, changed = forbidden_unchanged(case_dir, forbidden)
    score += _criterion(criteria, "forbidden_edits_unchanged", 0.10, forbidden_ok, changed)

    runner = select_runner()
    # Behavior-based gate: the submission is run as-is and measured against the
    # ExpectedMetrics tolerance bands (no dictionary string matching).
    repair_ok, repair_detail = family_repair_ok(case_dir, family, expected, runner)
    score += _criterion(criteria, "family_specific_repair_present", 0.10, repair_ok, repair_detail)

    variant_results = []
    all_mesh = all_solver = all_post = all_metrics = all_clean = True
    variants_applied = bool(hidden_variants)
    for variant in hidden_variants if repair_ok else []:
        tmp = Path(tempfile.mkdtemp(prefix="synthetic-openfoam-variant-"))
        try:
            variant_case = tmp / "case"
            shutil.copytree(case_dir, variant_case)
            apply_hidden_variant(variant_case, variant)
            variant_expected = dict(expected)
            variant_expected.update(variant.get("expected_overrides") or {})
            anchor = (metric_anchors.get("by_variant") or {}).get(str(variant.get("name")), {})
            if anchor:
                variant_expected["metric_anchor"] = anchor
            run = runner.run(variant_case, variant_expected, variant)
            variant_results.append({
                "name": variant.get("name"),
                "ok": run.ok,
                "mesh_ok": run.mesh_ok,
                "solver_ok": run.solver_ok,
                "post_ok": run.post_ok,
                "metrics_ok": run.metrics_ok,
                "no_numerical_errors": run.no_numerical_errors,
                "metrics": run.metrics,
                "reason": run.reason,
                "log_tails": {name: text[-2000:] for name, text in (run.logs or {}).items()},
                "postprocessing": run.postprocessing or {},
                "expected_overrides": variant.get("expected_overrides") or {},
                "metric_anchor_present": bool(anchor),
            })
            all_mesh = all_mesh and run.mesh_ok
            all_solver = all_solver and run.solver_ok
            all_post = all_post and run.post_ok
            all_metrics = all_metrics and run.metrics_ok
            all_clean = all_clean and run.no_numerical_errors
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if not repair_ok:
        all_mesh = all_solver = all_post = all_metrics = all_clean = False

    metadata["hidden_variants"] = variant_results
    score += _criterion(criteria, "hidden_variants_applied", 0.05, variants_applied, [v.get("name") for v in hidden_variants])
    score += _criterion(criteria, "mesh_generation_and_checkmesh", 0.10, all_mesh, variant_results)
    score += _criterion(criteria, "solver_reaches_end_time", 0.15, all_solver, variant_results)
    score += _criterion(criteria, "no_nan_fpe_or_fatal_errors", 0.10, all_clean, variant_results)
    score += _criterion(criteria, "required_postprocessing_outputs", 0.10, all_post, expected.get("postprocessing_required", []))
    score += _criterion(criteria, "engineering_metrics_within_tolerance", 0.20, all_metrics, expected)

    score = max(0.0, min(1.0, score))
    return _result(score, criteria, metadata)
