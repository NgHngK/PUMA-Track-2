from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_signature(path: Path, *, content_hash: bool = False) -> dict[str, Any]:
    path = path.resolve()
    stat = path.stat()
    result: dict[str, Any] = {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash:
        result["sha256"] = sha256_file(path)
    return result


def directory_metadata_signature(directory: Path, patterns: Iterable[str]) -> dict[str, Any]:
    directory = directory.resolve()
    rows: list[tuple[str, int, int]] = []
    for pattern in patterns:
        for path in directory.rglob(pattern):
            if path.is_file():
                stat = path.stat()
                rows.append((str(path.relative_to(directory)), int(stat.st_size), int(stat.st_mtime_ns)))
    rows = sorted(set(rows))
    payload = json.dumps(rows, separators=(",", ":"), ensure_ascii=False)
    return {
        "directory": str(directory),
        "files": len(rows),
        "metadata_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }
