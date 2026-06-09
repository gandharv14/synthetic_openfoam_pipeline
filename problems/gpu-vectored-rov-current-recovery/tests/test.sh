#!/usr/bin/env bash
set -euo pipefail

python -m py_compile scorer/compute_score.py

WORKSPACE="$(mktemp -d)"
LOG_DIR="${LBT_VERIFIER_DIR:-/logs/verifier}"
if ! mkdir -p "${LOG_DIR}" 2>/dev/null; then
  LOG_DIR="$(mktemp -d)"
fi
trap 'rm -rf "${WORKSPACE}"' EXIT

cat >"${WORKSPACE}/policy.py" <<'PY'
import numpy as np


class Policy:
    def act(self, obs):
        return np.zeros(8, dtype=float)
PY

python - <<'PY'
from pathlib import Path
import mujoco

model = mujoco.MjModel.from_xml_path(str(Path.cwd() / "data" / "rov_model.xml"))
assert model.nq == 7
assert model.nv == 6
assert model.nu == 8
PY

uv run python -m grader_runner.run_grader \
  --workspace "${WORKSPACE}" \
  --grader-dir scorer \
  --private-dir scorer/data \
  --output-dir "${LOG_DIR}"

python - <<'PY' "${LOG_DIR}"
import json
from pathlib import Path
import sys

log_dir = Path(sys.argv[1])
assert (log_dir / "reward.json").exists()
assert (log_dir / "reward-details.json").exists()
assert (log_dir / "reward.txt").exists()
assert json.loads((log_dir / "reward.json").read_text())["score"] == 0.0
PY
