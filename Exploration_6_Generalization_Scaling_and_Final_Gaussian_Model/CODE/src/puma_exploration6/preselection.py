from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from .constants import CLASSES, NUM_CLASSES
from .manifest import write_manifest
from .x3_builder import BASE_TRAIN_QUOTA, DEV_QUOTA, LOCKED_QUOTA


@dataclass(frozen=True)
class AdmissionAudit:
    historical_rois: tuple[str, ...]
    unused_rois: tuple[str, ...]
    unused_class_counts: tuple[int, ...]
    strict_locked_deficits: tuple[int, ...]
    strict_locked_feasible: bool


def read_historical_rois(path: str | Path) -> set[str]:
    """Read an authoritative historical ROI list/manifest.

    Supported formats:
      * CSV/TSV with an ``roi`` column
      * TXT with one ROI per line
      * JSON list of ROI strings
      * JSON object with ``rois`` / ``historical_rois`` list

    The function intentionally does NOT recursively scrape reports for ROI names:
    historical exposure must come from an explicit authoritative artifact.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open(newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if not reader.fieldnames or "roi" not in reader.fieldnames:
                raise ValueError(f"{path} must contain an 'roi' column")
            rois = {(row.get("roi") or "").strip() for row in reader}
    elif suffix == ".txt":
        rois = {x.strip() for x in path.read_text(encoding="utf-8").splitlines()}
    elif suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            rois = {str(x).strip() for x in obj}
        elif isinstance(obj, dict):
            values = obj.get("historical_rois", obj.get("rois"))
            if not isinstance(values, list):
                raise ValueError(f"{path} JSON requires a list or a rois/historical_rois list")
            rois = {str(x).strip() for x in values}
        else:
            raise ValueError(f"unsupported historical ROI JSON structure in {path}")
    else:
        raise ValueError(f"unsupported historical ROI file type: {path.suffix}")

    rois.discard("")
    if not rois:
        raise ValueError(f"no ROI identifiers found in {path}")
    return rois


def class_counts(rows: Sequence[dict]) -> np.ndarray:
    out = np.zeros(NUM_CLASSES, dtype=np.int64)
    for r in rows:
        y = int(r["label"])
        if not 0 <= y < NUM_CLASSES:
            raise ValueError(f"invalid label {y}")
        out[y] += 1
    return out


def audit_historical_overlap(rows: Sequence[dict], historical_rois: set[str]) -> AdmissionAudit:
    all_rois = {str(r["roi"]) for r in rows}
    unknown = historical_rois - all_rois
    if unknown:
        sample = sorted(unknown)[:10]
        raise ValueError(
            f"historical ROI artifact contains {len(unknown)} ROI(s) absent from the dataset; "
            f"examples={sample}"
        )
    unused = all_rois - historical_rois
    unused_rows = [r for r in rows if str(r["roi"]) in unused]
    counts = class_counts(unused_rows)
    deficits = np.maximum(np.asarray(LOCKED_QUOTA, dtype=np.int64) - counts, 0)
    return AdmissionAudit(
        historical_rois=tuple(sorted(historical_rois)),
        unused_rois=tuple(sorted(unused)),
        unused_class_counts=tuple(map(int, counts)),
        strict_locked_deficits=tuple(map(int, deficits)),
        strict_locked_feasible=bool(np.all(deficits == 0)),
    )


def _group_class_matrix(rows: Sequence[dict]) -> tuple[list[str], np.ndarray]:
    groups = sorted({str(r["group"]) for r in rows})
    gi = {g: i for i, g in enumerate(groups)}
    counts = np.zeros((len(groups), NUM_CLASSES), dtype=np.int64)
    for r in rows:
        counts[gi[str(r["group"])], int(r["label"])] += 1
    return groups, counts


def _milp_group_partition(
    rows: Sequence[dict],
    quotas: Sequence[Sequence[int]],
    *,
    seed: int = 17,
    min_groups_per_split: int = 1,
) -> list[set[str]]:
    """Find a leakage-free group partition with enough candidate nuclei.

    This is a feasibility problem, not a model-selection procedure.
    Each group is assigned to at most one split. Groups may remain unused.
    Class quotas are lower bounds on AVAILABLE candidates; exact nucleus
    selection is performed later with diversity-aware round-robin sampling.

    scipy.optimize.milp / HiGHS is deterministic for a fixed problem. Tiny
    seeded objective perturbations only choose between multiple feasible
    partitions without using labels from DEV/confirmation outcomes.
    """
    quotas = [np.asarray(q, dtype=np.float64) for q in quotas]
    if any(q.shape != (NUM_CLASSES,) for q in quotas):
        raise ValueError("every quota vector must have NUM_CLASSES entries")
    groups, gc = _group_class_matrix(rows)
    G, S = len(groups), len(quotas)
    if G < S * min_groups_per_split:
        raise ValueError("too few independent groups for requested partitions")

    # x[g,s] binary. y[g] binary means the group is used somewhere.
    n_x = G * S
    n_y = G
    n = n_x + n_y

    def xi(g: int, s: int) -> int:
        return g * S + s

    def yi(g: int) -> int:
        return n_x + g

    constraints = []
    lower = []
    upper = []

    # Each group goes to at most one split; y[g] == sum_s x[g,s].
    for g in range(G):
        row = np.zeros(n)
        for s in range(S):
            row[xi(g, s)] = 1
        constraints.append(row)
        lower.append(0)
        upper.append(1)

        row2 = row.copy()
        row2[yi(g)] = -1
        constraints.append(row2)
        lower.append(0)
        upper.append(0)

    # Every split must have enough available nuclei for every class.
    for s, q in enumerate(quotas):
        for k in range(NUM_CLASSES):
            row = np.zeros(n)
            for g in range(G):
                row[xi(g, s)] = gc[g, k]
            constraints.append(row)
            lower.append(float(q[k]))
            upper.append(np.inf)

    # Require at least a modest number of groups in every split.
    for s in range(S):
        row = np.zeros(n)
        for g in range(G):
            row[xi(g, s)] = 1
        constraints.append(row)
        lower.append(float(min_groups_per_split))
        upper.append(np.inf)

    A = np.asarray(constraints, dtype=np.float64)
    rng = np.random.default_rng(seed)

    # Maximize number of used groups (negative coefficient under minimization),
    # with tiny deterministic tie-breaking noise on split assignments.
    c = np.zeros(n, dtype=np.float64)
    c[:n_x] = rng.uniform(0.0, 1e-6, size=n_x)
    c[n_x:] = -1.0

    result = milp(
        c=c,
        integrality=np.ones(n, dtype=np.int8),
        bounds=Bounds(np.zeros(n), np.ones(n)),
        constraints=LinearConstraint(A, np.asarray(lower), np.asarray(upper)),
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise ValueError(
            "No leakage-free group partition satisfies the requested class quotas. "
            f"HiGHS status={result.status}, message={result.message}"
        )

    x = result.x[:n_x].reshape(G, S)
    parts = [set() for _ in range(S)]
    for g, name in enumerate(groups):
        assigned = np.flatnonzero(x[g] > 0.5)
        if len(assigned) > 1:
            raise AssertionError("MILP returned group leakage")
        if len(assigned) == 1:
            parts[int(assigned[0])].add(name)

    for i, p in enumerate(parts):
        if len(p) < min_groups_per_split:
            raise AssertionError(f"split {i} has insufficient groups")
    if any(parts[i] & parts[j] for i in range(S) for j in range(i + 1, S)):
        raise AssertionError("group leakage after MILP")
    return parts


def _diversity_order(
    rows: Sequence[dict],
    class_id: int,
    groups: set[str],
    *,
    seed: int,
) -> list[int]:
    by_group: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if int(r["label"]) == class_id and str(r["group"]) in groups:
            by_group[str(r["group"])].append(i)
    if not by_group:
        return []

    rng = np.random.default_rng(seed + 1009 * class_id)
    group_names = sorted(by_group)
    rng.shuffle(group_names)
    for g in group_names:
        rng.shuffle(by_group[g])

    # Round-robin across groups: first maximize independent-group diversity,
    # only then take second/third nuclei from the same group.
    out: list[int] = []
    depth = 0
    while True:
        added = False
        for g in group_names:
            if depth < len(by_group[g]):
                out.append(by_group[g][depth])
                added = True
        if not added:
            break
        depth += 1
    return out


def _select_exact(
    rows: Sequence[dict],
    groups: set[str],
    quotas: Sequence[int],
    *,
    seed: int,
) -> tuple[list[int], dict[int, list[int]]]:
    chosen: list[int] = []
    orders: dict[int, list[int]] = {}
    for k, q in enumerate(quotas):
        order = _diversity_order(rows, k, groups, seed=seed)
        orders[k] = order
        if len(order) < int(q):
            raise ValueError(
                f"insufficient {CLASSES[k]} candidates: need {q}, have {len(order)}"
            )
        chosen.extend(order[: int(q)])
    if len(chosen) != sum(map(int, quotas)):
        raise AssertionError("exact selection size mismatch")
    if len(set(chosen)) != len(chosen):
        raise AssertionError("duplicate nucleus selected")
    return chosen, orders


def _copy_with_split(row: dict, split: str) -> dict:
    out = dict(row)
    out["split"] = split
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _support_rows(rows: Sequence[dict], roi_set: set[str], cohort: str) -> list[dict]:
    selected = [r for r in rows if str(r["roi"]) in roi_set]
    roi_class: dict[tuple[str, int], int] = Counter(
        (str(r["roi"]), int(r["label"])) for r in selected
    )
    out = []
    for roi in sorted(roi_set):
        roi_rows = [r for r in selected if str(r["roi"]) == roi]
        d = {
            "cohort": cohort,
            "roi": roi,
            "nuclei": len(roi_rows),
            "v17_components": sum(
                len(json.loads(r.get("eval_centroids", "[]") or "[]")) for r in roi_rows
            ),
        }
        for k, name in enumerate(CLASSES):
            d[name] = roi_class.get((roi, k), 0)
        out.append(d)
    return out


def preselect_prompt6(
    rows: list[dict],
    historical_rois: set[str],
    out_dir: str | Path,
    *,
    seed: int = 17,
    min_groups_per_split: int = 5,
) -> dict:
    """Create a Exploration-6-ready preselection without touching image pixels.

    Current recommended protocol when strict historically-untouched 10-class
    LOCKED is impossible:
      * reserve ALL historically unused ROIs as LOCKED_NATURAL;
      * build D300/D600/D900 + DEV225 + PROMPT6_CONFIRMATION225 from the
        historically exposed ROI pool, with group-disjoint partitions;
      * label confirmation honestly as prospectively locked but historically
        exposed.

    The function only selects manifest rows. It never copies or modifies TIFF/
    GeoJSON source files.
    """
    out_dir = Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; preserve it and choose a new path: {out_dir}"
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    if len({str(r["uid"]) for r in rows}) != len(rows):
        raise ValueError("full manifest contains duplicate UIDs")

    audit = audit_historical_overlap(rows, historical_rois)
    all_rois = {str(r["roi"]) for r in rows}
    unused_rois = set(audit.unused_rois)
    research_rois = all_rois - unused_rois
    research_rows = [r for r in rows if str(r["roi"]) in research_rois]

    # Preserve the actual historically-unused pool exactly; do not balance it.
    locked_natural = [
        _copy_with_split(r, "locked_natural")
        for r in rows
        if str(r["roi"]) in unused_rois
    ]
    write_manifest(
        out_dir / "LOCKED_NATURAL.csv",
        locked_natural,
        {
            "cohort": "LOCKED_NATURAL",
            "historically_unused": True,
            "class_balanced": False,
            "subsampled": False,
            "note": "All historically unused ROIs; zero-support class metrics must be reported as not estimable.",
        },
    )

    # Build the all-ten-class Exploration-6 research partitions only from the
    # historically exposed pool. The third partition is confirmation, not a
    # historically untouched test.
    quotas = [
        [3 * q for q in BASE_TRAIN_QUOTA],  # 900
        list(DEV_QUOTA),                     # 225
        list(LOCKED_QUOTA),                  # 225 -> Exploration6 confirmation
    ]
    train_groups, dev_groups, confirmation_groups = _milp_group_partition(
        research_rows,
        quotas,
        seed=seed,
        min_groups_per_split=min_groups_per_split,
    )

    train_idx, train_orders = _select_exact(
        research_rows, train_groups, quotas[0], seed=seed
    )
    dev_idx, _ = _select_exact(
        research_rows, dev_groups, quotas[1], seed=seed + 1
    )
    conf_idx, _ = _select_exact(
        research_rows, confirmation_groups, quotas[2], seed=seed + 2
    )

    selected_uids = {
        str(research_rows[i]["uid"]) for i in train_idx + dev_idx + conf_idx
    }
    if len(selected_uids) != 1350:
        raise AssertionError("research cohort must contain exactly 1,350 unique nuclei")

    # D300 ⊂ D600 ⊂ D900, with fixed DEV and confirmation.
    manifests: dict[str, str] = {}
    for multiplier, name in ((1, "D300"), (2, "D600"), (3, "D900")):
        tr: list[int] = []
        for k, base_q in enumerate(BASE_TRAIN_QUOTA):
            tr.extend(train_orders[k][: multiplier * int(base_q)])
        manifest_rows = (
            [_copy_with_split(research_rows[i], "train") for i in tr]
            + [_copy_with_split(research_rows[i], "dev") for i in dev_idx]
            + [_copy_with_split(research_rows[i], "locked") for i in conf_idx]
        )
        p = out_dir / f"{name}.csv"
        write_manifest(
            p,
            manifest_rows,
            {
                "dataset": name,
                "seed": seed,
                "group_disjoint": True,
                "locked_semantics": (
                    "PROMPT6_CONFIRMATION: prospectively locked for Exploration 6, "
                    "but historically exposed at ROI level"
                ),
                "locked_natural_manifest": "LOCKED_NATURAL.csv",
            },
        )
        manifests[name] = str(p)

    # Human-readable and machine-readable census.
    support_rows = (
        _support_rows(rows, research_rois, "historical_research_pool")
        + _support_rows(rows, unused_rois, "locked_natural")
    )
    _write_csv(out_dir / "ROI_CENSUS.csv", support_rows)

    hist_counts = class_counts(research_rows)
    unused_counts = np.asarray(audit.unused_class_counts)
    class_table = []
    for k, name in enumerate(CLASSES):
        class_table.append(
            {
                "class_id": k,
                "class_name": name,
                "full_dataset": int(hist_counts[k] + unused_counts[k]),
                "historical_research_pool": int(hist_counts[k]),
                "locked_natural": int(unused_counts[k]),
                "strict_original_locked_target": int(LOCKED_QUOTA[k]),
                "strict_original_locked_deficit": int(audit.strict_locked_deficits[k]),
            }
        )
    _write_csv(out_dir / "CLASS_CENSUS.csv", class_table)

    _write_csv(
        out_dir / "HISTORICAL_ROIS.csv",
        [{"roi": x} for x in sorted(research_rois)],
    )
    _write_csv(
        out_dir / "HISTORICALLY_UNUSED_ROIS.csv",
        [{"roi": x} for x in sorted(unused_rois)],
    )

    group_table = []
    for split_name, groups in (
        ("train", train_groups),
        ("dev", dev_groups),
        ("prompt6_confirmation", confirmation_groups),
    ):
        for g in sorted(groups):
            group_table.append({"split": split_name, "group": g})
    _write_csv(out_dir / "RESEARCH_GROUP_ASSIGNMENT.csv", group_table)

    summary = {
        "dataset_rois": len(all_rois),
        "dataset_nuclei": len(rows),
        "historical_rois": len(research_rois),
        "historically_unused_rois": len(unused_rois),
        "locked_natural_nuclei": len(locked_natural),
        "locked_natural_class_counts": {
            CLASSES[k]: int(unused_counts[k]) for k in range(NUM_CLASSES)
        },
        "strict_original_locked_quota": {
            CLASSES[k]: int(LOCKED_QUOTA[k]) for k in range(NUM_CLASSES)
        },
        "strict_original_locked_deficits": {
            CLASSES[k]: int(audit.strict_locked_deficits[k]) for k in range(NUM_CLASSES)
        },
        "strict_original_locked_feasible": audit.strict_locked_feasible,
        "protocol_used": "LOCKED_NATURAL + prospectively_locked_prompt6_confirmation",
        "research_train": 900,
        "research_dev": 225,
        "prompt6_confirmation": 225,
        "research_partition_groups": {
            "train": len(train_groups),
            "dev": len(dev_groups),
            "prompt6_confirmation": len(confirmation_groups),
        },
        "manifests": manifests,
        "locked_natural_manifest": str(out_dir / "LOCKED_NATURAL.csv"),
        "warning": (
            None
            if audit.strict_locked_feasible
            else (
                "The same source dataset cannot create the originally requested "
                "historically untouched balanced 10-class LOCKED cohort. "
                "LOCKED_NATURAL is preserved separately and Exploration6 confirmation "
                "is explicitly labeled historically exposed."
            )
        ),
    }
    (out_dir / "PRESELECTION_SUMMARY.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    lines = [
        "# Exploration 6 dataset preselection",
        "",
        f"- Dataset ROIs: **{summary['dataset_rois']}**",
        f"- Dataset nuclei: **{summary['dataset_nuclei']}**",
        f"- Historical ROIs: **{summary['historical_rois']}**",
        f"- Historically unused ROIs: **{summary['historically_unused_rois']}**",
        f"- LOCKED_NATURAL nuclei: **{summary['locked_natural_nuclei']}**",
        f"- Strict original untouched LOCKED feasible: **{summary['strict_original_locked_feasible']}**",
        "",
        "## Historically unused class support",
        "",
        "| class | available | original target | deficit |",
        "|---|---:|---:|---:|",
    ]
    for k, name in enumerate(CLASSES):
        lines.append(
            f"| {name} | {unused_counts[k]} | {LOCKED_QUOTA[k]} | "
            f"{audit.strict_locked_deficits[k]} |"
        )
    lines += [
        "",
        "## Recommended use",
        "",
        "- `D300.csv`, `D600.csv`, `D900.csv`: Exploration-6 research manifests.",
        "- Their `locked` split is **PROMPT6_CONFIRMATION**, not historically untouched.",
        "- `LOCKED_NATURAL.csv`: every historically unused ROI, never balanced/subsampled.",
        "- Broad HPO remains TRAIN-only grouped CV.",
        "- DEV selects finalists.",
        "- PROMPT6_CONFIRMATION is opened only after choices are frozen.",
        "- LOCKED_NATURAL is opened once for final natural unseen-ROI diagnostics.",
        "",
        "No TIFF or GeoJSON file is copied or modified.",
    ]
    (out_dir / "PRESELECTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
