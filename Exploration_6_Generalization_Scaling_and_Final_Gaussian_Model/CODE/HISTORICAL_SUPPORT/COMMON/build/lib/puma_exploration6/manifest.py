from __future__ import annotations

import csv
import json
import hashlib
from pathlib import Path
from typing import Iterable
import numpy as np

from .constants import CLASSES, NUM_CLASSES

REQUIRED_COLUMNS = {"uid","roi","group","image","x","y","label","split","coordinate_source"}
VALID_SPLITS = {"train","dev","val","calibration","locked","test","predict"}
VALID_COORDINATE_SOURCES = {"gt","stage1_oof","stage1_frozen"}

ROW_IDENTITY_FIELDS = ("uid","roi","group","image","x","y","label","coordinate_source")

def row_identity_sha256(rows: list[dict]) -> str:
    """Hash ordered sample identity while intentionally excluding split assignment.

    This lets immutable image/Tier-A/UNI2 caches be reused across internal grouped-CV
    manifests that contain the exact same ordered nuclei but relabel TRAIN groups into
    temporary train/dev folds. Any change to row order, image, point, label, group, or
    coordinate source changes the hash and fails closed.
    """
    payload=[]
    for r in rows:
        payload.append({k:r.get(k) for k in ROW_IDENTITY_FIELDS})
    blob=json.dumps(payload,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def read_manifest(path: str | Path, require_files: bool = True) -> list[dict]:
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("manifest is empty")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"manifest missing columns: {sorted(missing)}")
    if "class_name" not in rows[0]:
        sidecar = path.with_suffix(".json")
        if not sidecar.exists() or tuple(json.loads(sidecar.read_text(encoding="utf-8")).get("classes", ())) != CLASSES:
            raise ValueError("manifest requires class_name or a sidecar with canonical classes")

    seen_uid: set[str] = set()
    ledgers: dict[str, dict[str,str]] = {"group":{}, "roi":{}, "image":{}}
    resolved: dict[str,str] = {}
    for row in rows:
        uid = row["uid"].strip()
        if not uid or uid in seen_uid:
            raise ValueError(f"empty or duplicate uid: {uid!r}")
        seen_uid.add(uid); row["uid"] = uid
        split = row["split"].strip()
        if split == "val": split = "dev"
        if split not in VALID_SPLITS:
            raise ValueError(f"unknown split: {split}")
        row["split"] = split
        src = row["coordinate_source"].strip()
        if src not in VALID_COORDINATE_SOURCES:
            raise ValueError(f"unknown coordinate_source: {src}")
        row["coordinate_source"] = src
        raw_image = row["image"].strip()
        if not raw_image:
            raise ValueError(f"empty image path for {uid}")
        if raw_image not in resolved:
            p = Path(raw_image)
            p = p if p.is_absolute() else path.parent / p
            resolved[raw_image] = str(p.resolve())
            if require_files and not Path(resolved[raw_image]).is_file():
                raise FileNotFoundError(resolved[raw_image])
        row["image"] = resolved[raw_image]
        for key in ("group","roi","image"):
            raw = row[key].strip() if key != "image" else row[key]
            if not raw:
                raise ValueError(f"empty {key} for {uid}")
            ledger_key = str(Path(raw)).casefold() if key == "image" else raw
            old = ledgers[key].get(ledger_key)
            if old is not None and old != split:
                raise ValueError(f"{key} leakage across splits: {raw}: {old} vs {split}")
            ledgers[key][ledger_key] = split
        try:
            label = int(row["label"]); x = float(row["x"]); y = float(row["y"])
        except Exception as e:
            raise ValueError(f"invalid numeric fields for {uid}") from e
        if label not in range(NUM_CLASSES) and not (split == "predict" and label == -1):
            raise ValueError(f"label outside canonical ontology for {uid}: {label}")
        if not np.isfinite([x,y]).all():
            raise ValueError(f"nonfinite coordinates for {uid}")
        row["label"] = label; row["x"] = x; row["y"] = y
        if label >= 0 and "class_name" in row and row["class_name"].strip() != CLASSES[label]:
            raise ValueError(f"class_name mismatch for {uid}")
        for k in ("eval_x","eval_y"):
            if k in row and row[k] not in (None, ""):
                row[k] = float(row[k])
    return rows


def write_manifest(path: str | Path, rows: Iterable[dict], metadata: dict | None = None) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError("cannot write empty manifest")
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields: fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    sidecar = {"classes": list(CLASSES)}
    if metadata: sidecar.update(metadata)
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")


def split_rows(rows: list[dict], split: str) -> list[dict]:
    split = "dev" if split == "val" else split
    return [r for r in rows if r["split"] == split]


def assert_disjoint(rows: list[dict], keys=("group","roi","image")) -> None:
    for key in keys:
        by: dict[str,set[str]] = {}
        for r in rows: by.setdefault(str(r[key]), set()).add(r["split"])
        leaks = {k:v for k,v in by.items() if len(v)>1}
        if leaks:
            first = next(iter(leaks.items()))
            raise ValueError(f"{key} leakage: {first}")
