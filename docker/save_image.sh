#!/usr/bin/env bash
set -euo pipefail
OUT="${1:-puma_pipeline.tar.gz}"
docker save puma:latest | gzip -1 > "$OUT"
ls -lh "$OUT"
