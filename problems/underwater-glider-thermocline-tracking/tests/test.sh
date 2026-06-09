#!/usr/bin/env bash
set -euo pipefail

python -m py_compile data/glider_env.py scorer/compute_score.py solution/render_config.py
bash -n solution/solve.sh
bash -n solution/render.sh
