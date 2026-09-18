#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TEST_INPUT="${1:?Usage: test_container.sh /path/to/1024x1024.tif}"
OUT="${2:-$ROOT/docker_test_output}"
rm -rf "$OUT" && mkdir -p "$OUT/input/images/melanoma-whole-slide-image" "$OUT/output"
cp "$TEST_INPUT" "$OUT/input/images/melanoma-whole-slide-image/$(basename "$TEST_INPUT")"
docker run --rm --gpus all --network none --memory 32g \
  -v "$OUT/input:/input:ro" -v "$OUT/output:/output" puma:latest
python "$ROOT/scripts/validate_submission.py" "$OUT/output"
