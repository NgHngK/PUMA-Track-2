#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
required=(
  stage1_final_ema.pt
  stage2_final_ema.pt
  tissue_final_ema.pt
  risk_calibration.npz
  uni2_h_model.bin
  model_manifest.json
)
for name in "${required[@]}"; do
  if [[ ! -f "$ROOT/models/$name" ]]; then
    echo "ERROR: missing $ROOT/models/$name" >&2
    echo "Run: python $ROOT/scripts/package_deployment.py --config <resolved-config.json>" >&2
    exit 2
  fi
done
docker build -f "$ROOT/docker/Dockerfile" -t puma:latest "$ROOT"
