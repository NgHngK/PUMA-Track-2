from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from puma_pipeline.submission.format import validate_puma_nuclei_json, validate_tissue_tiff


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate a PUMA Track-2 output directory before submission."
    )
    parser.add_argument(
        "output", nargs="?", default="/output",
        help="Output directory containing nuclei JSON and tissue-mask TIFF (default: /output).",
    )
    args = parser.parse_args()

    output = Path(args.output)
    nuclei = output / "melanoma-10-class-nuclei-segmentation.json"
    tissue_dir = output / "images" / "melanoma-tissue-mask-segmentation"
    if not nuclei.is_file():
        raise FileNotFoundError(nuclei)
    payload = json.loads(nuclei.read_text(encoding="utf-8"))
    validate_puma_nuclei_json(payload)

    tissue = sorted(tissue_dir.glob("*.tif")) + sorted(tissue_dir.glob("*.tiff"))
    if len(tissue) != 1:
        raise RuntimeError(f"Expected exactly one tissue TIFF, got {tissue}")
    mask = validate_tissue_tiff(tissue[0])
    print(
        f"VALID: {len(payload['polygons'])} nuclei; "
        f"tissue={tissue[0].name}; labels={sorted(np.unique(mask).tolist())}"
    )


if __name__ == "__main__":
    main()
