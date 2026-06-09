# Synthetic OpenFOAM Pipeline

Root-level generator for synthetic OpenFOAM RL tasks. The pipeline source stays
outside `lbx-rl-tasks-iso-template`, but generated problems are authored directly
under `lbx-rl-tasks-iso-template/problems/` by default so the template harness can
build, validate, and run local agent attempts in place. Each exported problem is
also mirrored to `synthetic_openfoam_pipeline/problems/` for inspection and
pipeline-local artifacts without changing the harness source of truth.

## Common Commands

```bash
python -m synthetic_openfoam_pipeline --help
python -m synthetic_openfoam_pipeline generate --count 5 --template-root lbx-rl-tasks-iso-template --output-root generated_openfoam_tasks --seed 123
python -m synthetic_openfoam_pipeline validate --tasks-root lbx-rl-tasks-iso-template/problems
python -m synthetic_openfoam_pipeline deploy-handful --count 5 --template-root lbx-rl-tasks-iso-template --output-root generated_openfoam_tasks --seed 123 --target claude-harness --attempts 3 --harness-timeout-sec 300 --zip
```

The generated tasks follow the OpenFOAM example layout from the template repo:
`task.toml`, `metadata.json`, `instruction.md`, `environment/Dockerfile`,
public `data/`, private `scorer/data/`, deterministic `scorer/compute_score.py`,
oracle `solution/solve.sh`, weak `baselines/naive.sh`, and `tests/test.sh`.

By default, the CLI asks an Anthropic model to rewrite each generated
`instruction.md` and task description into a more natural human repair ticket
while preserving the submission contract and public constraints. Export
`ANTHROPIC_API_KEY` first, or source the root `.env.local` before generation:

```bash
set -a && source .env.local && set +a
python -m synthetic_openfoam_pipeline generate --count 5 --template-root lbx-rl-tasks-iso-template --output-root generated_openfoam_tasks --seed 123
```

Use `--prompt-model` to choose a model, or `--no-prompt-rewrite` for fully
deterministic text generation without an LLM call.

## Validation

The generated scorer always runs real OpenFOAM commands: `blockMesh`,
`checkMesh`, and `simpleFoam`. Pipeline validation shells out to the template
harness ground-truth runtime so validation happens inside the task image, where
OpenFOAM is provided by `/etc/solver-envs.d/openfoam.sh`.

Claude filtering defaults to the template harness target. From the template root
the pipeline runs:

```bash
set -a && source ../.env.local && set +a
uv run lbx-rl-harness run --problem-dir problems/<task_id> --runtime agent
```

The key is sourced only inside the subprocess and is not written to reports.
Missing credentials mark tasks as `needs_claude_validation` rather than accepted.
Reports and zips remain under root `generated_openfoam_tasks/`.

