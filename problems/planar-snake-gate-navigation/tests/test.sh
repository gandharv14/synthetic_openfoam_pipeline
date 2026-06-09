#!/usr/bin/env bash
set -euo pipefail

python -m py_compile data/snake_env.py scorer/compute_score.py solution/render_config.py
python - <<'PY'
import json
import tomllib
from pathlib import Path

base = Path(".")
tomllib.loads((base / "task.toml").read_text())
json.loads((base / "metadata.json").read_text())
json.loads((base / "data/public_scenarios.json").read_text())
json.loads((base / "scorer/data/hidden_scenarios.json").read_text())
print("static_parse_ok")
PY

tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT
cat > "$tmpdir/policy.py" <<'PY'
def act(obs):
    raise RuntimeError("intentional policy failure")
PY

POLICY_TMP="$tmpdir" python - <<'PY'
import os
from pathlib import Path

from scorer.compute_score import compute_score

result = compute_score(Path(os.environ["POLICY_TMP"]), None, Path("scorer/data"))
assert result["score"] == 0.0, result
diagnostics = result["metadata"]["diagnostics"]
assert diagnostics["finite_mean"] == 0.0, diagnostics
assert diagnostics["passed_gates_mean"] == 0.0, diagnostics
assert diagnostics["min_no_go_clearance_min"] <= 0.0, diagnostics
print("failed_policy_score_ok")
PY
