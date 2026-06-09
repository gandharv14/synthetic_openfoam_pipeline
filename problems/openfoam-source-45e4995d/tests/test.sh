#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PRIVATE_DIR="${LBT_PRIVATE_DIR:-/mcp_server/data}"
if [[ ! -d "${PRIVATE_DIR}" ]]; then
  PRIVATE_DIR="${TASK_DIR}/scorer/data"
fi
mkdir -p /tmp/openfoam-test-logs
python3 - "${TASK_DIR}" "${PRIVATE_DIR}" <<'PY'
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "scorer"))
from compute_score import compute_score
result = compute_score(Path("/tmp/output"), None, Path(sys.argv[2]))
Path("/tmp/openfoam-test-logs/reward.json").write_text(json.dumps(result, indent=2))
print("score:", result.get("score"))
PY
