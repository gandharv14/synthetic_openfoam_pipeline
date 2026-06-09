from __future__ import annotations

import argparse
import json
from pathlib import Path

from synthetic_openfoam_pipeline.agents.adversarial_filter_agent import AdversarialFilterAgent
from synthetic_openfoam_pipeline.agents.robustness_verifier_agent import RobustnessVerifierAgent
from synthetic_openfoam_pipeline.config import DEFAULT_FAMILIES, PipelineConfig, load_config
from synthetic_openfoam_pipeline.deployment.deploy_handful import deploy_handful, export_candidates, generate_candidates
from synthetic_openfoam_pipeline.export.reports import ReportWriter
from synthetic_openfoam_pipeline.rendering import render_all, update_report


def _families(value: str | None) -> list[str]:
    if not value:
        return list(DEFAULT_FAMILIES)
    return [part.strip() for part in value.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="synthetic_openfoam_pipeline")
    parser.add_argument("--config", help="Optional JSON/YAML config file")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate task directories")
    gen.add_argument("--count", type=int, default=5)
    gen.add_argument("--output-root", default="generated_openfoam_tasks")
    gen.add_argument("--template-root", default="lbx-rl-tasks-iso-template")
    gen.add_argument("--problems-root")
    gen.add_argument("--mirror-problems-root")
    gen.add_argument("--no-mirror-problems", action="store_true")
    gen.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    gen.add_argument("--seed", type=int, default=123)
    gen.add_argument("--zip", action="store_true")
    gen.add_argument("--prompt-model", help="Anthropic model for LLM prompt rewriting")
    gen.add_argument("--no-prompt-rewrite", action="store_true", help="Disable LLM prompt paraphrasing")

    val = sub.add_parser("validate", help="Validate generated task directories")
    val.add_argument("--tasks-root", default="lbx-rl-tasks-iso-template/problems")
    val.add_argument("--output-root", default="generated_openfoam_tasks")
    val.add_argument("--template-root", default="lbx-rl-tasks-iso-template")

    adv = sub.add_parser("adversarial", help="Run adversarial filtering")
    adv.add_argument("--tasks-root", default="lbx-rl-tasks-iso-template/problems")
    adv.add_argument("--output-root", default="generated_openfoam_tasks")
    adv.add_argument("--template-root", default="lbx-rl-tasks-iso-template")
    adv.add_argument("--target", default="claude-harness")
    adv.add_argument("--attempts", type=int, default=3)
    adv.add_argument("--harness-timeout-sec", type=int, default=300)

    dep = sub.add_parser("deploy-handful", help="Generate, validate, filter, and optionally zip tasks")
    dep.add_argument("--count", type=int, default=5)
    dep.add_argument("--output-root", default="generated_openfoam_tasks")
    dep.add_argument("--template-root", default="lbx-rl-tasks-iso-template")
    dep.add_argument("--problems-root")
    dep.add_argument("--mirror-problems-root")
    dep.add_argument("--no-mirror-problems", action="store_true")
    dep.add_argument("--families", default=",".join(DEFAULT_FAMILIES))
    dep.add_argument("--seed", type=int, default=123)
    dep.add_argument("--target", default="claude-harness")
    dep.add_argument("--attempts", type=int, default=3)
    dep.add_argument("--harness-timeout-sec", type=int, default=300)
    dep.add_argument("--zip", action="store_true")
    dep.add_argument("--skip-harness-ground-truth", action="store_true")
    dep.add_argument("--prompt-model", help="Anthropic model for LLM prompt rewriting")
    dep.add_argument("--no-prompt-rewrite", action="store_true", help="Disable LLM prompt paraphrasing")

    render = sub.add_parser("render-report", help="Render OpenFOAM review videos and embed them in the HTML report")
    render.add_argument("--tasks-root", default="lbx-rl-tasks-iso-template/problems")
    render.add_argument("--assets-dir", default="synthetic_openfoam_pipeline/renders")
    render.add_argument("--report", default="synthetic_openfoam_problem_report.html")
    render.add_argument("--frames", type=int, default=36)
    render.add_argument("--fps", type=int, default=12)
    return parser


def _problem_config(args) -> dict:
    problems_root = Path(args.problems_root) if getattr(args, "problems_root", None) else None
    mirror_problems_root = Path(args.mirror_problems_root) if getattr(args, "mirror_problems_root", None) else None
    return {
        "template_root": Path(args.template_root),
        "problems_root_override": problems_root,
        "mirror_problems": not getattr(args, "no_mirror_problems", False),
        "mirror_problems_root_override": mirror_problems_root,
        "use_template_problems": problems_root is None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "generate":
        config = load_config(
            args.config,
            count=args.count,
            output_root=Path(args.output_root),
            **_problem_config(args),
            families=_families(args.families),
            seed=args.seed,
            zip_tasks=args.zip,
            prompt_rewrite_model=args.prompt_model,
            prompt_rewrite_enabled=not args.no_prompt_rewrite,
        )
        candidates = generate_candidates(config)
        export_candidates(config, candidates)
        ReportWriter(config).write(candidates)
        print(json.dumps({"generated": [c.task_id for c in candidates], "output_root": str(config.output_root)}, indent=2))
        return 0

    if args.command == "validate":
        tasks_root = Path(args.tasks_root)
        config = load_config(
            args.config,
            output_root=Path(args.output_root),
            template_root=Path(args.template_root),
            problems_root_override=tasks_root,
            use_template_problems=False,
        )
        results = RobustnessVerifierAgent(config).validate_many(tasks_root)
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return 0 if all(result.status.value != "rejected" for result in results) else 1

    if args.command == "adversarial":
        tasks_root = Path(args.tasks_root)
        config = load_config(
            args.config,
            output_root=Path(args.output_root),
            template_root=Path(args.template_root),
            problems_root_override=tasks_root,
            use_template_problems=False,
            target=args.target,
            attempts=args.attempts,
            harness_timeout_sec=args.harness_timeout_sec,
        )
        validations = RobustnessVerifierAgent(config).validate_many(tasks_root)
        results = AdversarialFilterAgent(config).run_many(tasks_root, validations, args.target)
        print(json.dumps([result.to_dict() for result in results], indent=2))
        return 0 if all(result.status.value != "rejected" for result in results) else 1

    if args.command == "deploy-handful":
        config = load_config(
            args.config,
            count=args.count,
            output_root=Path(args.output_root),
            **_problem_config(args),
            families=_families(args.families),
            seed=args.seed,
            target=args.target,
            attempts=args.attempts,
            harness_timeout_sec=args.harness_timeout_sec,
            zip_tasks=args.zip,
            skip_harness_ground_truth=args.skip_harness_ground_truth,
            prompt_rewrite_model=args.prompt_model,
            prompt_rewrite_enabled=not args.no_prompt_rewrite,
        )
        result = deploy_handful(config)
        print(
            json.dumps(
                {
                    "tasks": result.task_ids,
                    "output_root": str(config.output_root),
                    "statuses": {c.task_id: c.status.value for c in result.candidates},
                },
                indent=2,
            )
        )
        return 0 if all(c.status.value != "rejected" for c in result.candidates) else 1

    if args.command == "render-report":
        renders = render_all(Path(args.tasks_root), Path(args.assets_dir), frames=args.frames, fps=args.fps)
        update_report(Path(args.report), renders, Path(args.assets_dir))
        print(
            json.dumps(
                {
                    "rendered": [str(render.path) for render in renders],
                    "report": str(Path(args.report)),
                },
                indent=2,
            )
        )
        return 0

    parser.error("unhandled command")
    return 2

