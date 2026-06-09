#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUTPUT_DIR}"

if [[ "$(uname -s)" != "Darwin" ]]; then
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
else
  unset MUJOCO_GL
  unset PYOPENGL_PLATFORM
fi

if [ ! -f "${OUTPUT_DIR}/policy.py" ] || [ ! -f "${OUTPUT_DIR}/model.xml" ]; then
  LBT_OUTPUT_DIR="${OUTPUT_DIR}" bash solution/solve.sh
fi

uv run python -m lbx_rl_tasks_harness.render_mujoco \
  --model "${OUTPUT_DIR}/model.xml" \
  --policy "${OUTPUT_DIR}/policy.py" \
  --output "${OUTPUT_DIR}/rendering.mp4" \
  --config solution/render_config.py \
  --duration-sec 10.0
