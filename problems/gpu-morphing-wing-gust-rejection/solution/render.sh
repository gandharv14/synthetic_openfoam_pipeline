#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${LBT_OUTPUT_DIR:-/tmp/output}"
MODEL_PATH="${MODEL_PATH:-data/morphing_wing.xml}"

if [[ -x /usr/bin/ffmpeg ]]; then
  export PATH="/usr/bin:${PATH}"
fi

uv run python -m lbx_rl_tasks_harness.render_mujoco \
  --model "${MODEL_PATH}" \
  --policy "${OUTPUT_DIR}/policy.py" \
  --output "${OUTPUT_DIR}/rendering.mp4" \
  --config solution/render_config.py \
  --duration-sec 7.0 \
  --width 1280 \
  --height 720
