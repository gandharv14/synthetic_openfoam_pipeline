#!/usr/bin/env bash
set -euo pipefail
OUT="${LBT_OUTPUT_DIR:-/tmp/output}"
mkdir -p "${OUT}"
ffmpeg -y -f lavfi -i color=c=navy:s=1280x720:d=1 -pix_fmt yuv420p "${OUT}/rendering.mp4" >/dev/null 2>&1
