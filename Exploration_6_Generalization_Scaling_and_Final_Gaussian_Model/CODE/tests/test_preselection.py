from pathlib import Path
import csv
import numpy as np

from puma_exploration6.constants import CLASSES
from puma_exploration6.preselection import (
    read_historical_rois,
    audit_historical_overlap,
    _milp_group_partition,
    _select_exact,
)


def _row(uid, roi, group, label):
    return {
        "uid": uid, "roi": roi, "group": group, "image": "x.tif",
        "x": 1.0, "y": 1.0, "eval_x": 1.0, "eval_y": 1.0,
        "eval_centroids": "[[1.0,1.0]]", "label": label,
        "class_name": CLASSES[label], "split": "train",
        "coordinate_source": "gt",
    }


def test_read_historical_csv(tmp_path):
    p = tmp_path / "h.csv"
    p.write_text("roi\nR1\nR2\nR1\n", encoding="utf-8")
    assert read_historical_rois(p) == {"R1", "R2"}


def test_audit_strict_deficit():
    rows = []
    # R1 historical; R2 unused has no class 0 except one class 1.
    for k in range(10):
        rows.append(_row(f"R1:{k}", "R1", "R1", k))
    rows.append(_row("R2:0", "R2", "R2", 1))
    a = audit_historical_overlap(rows, {"R1"})
    assert a.unused_rois == ("R2",)
    assert not a.strict_locked_feasible
    assert a.unused_class_counts[1] == 1
    assert a.strict_locked_deficits[0] > 0


def test_milp_partition_and_exact_selection():
    rows = []
    # 12 groups, each deliberately has generous support for every class.
    for g in range(12):
        for k in range(10):
            for j in range(8):
                rows.append(_row(f"G{g}:{k}:{j}", f"R{g}", f"G{g}", k))
    quotas = [[8]*10, [8]*10, [8]*10]
    parts = _milp_group_partition(rows, quotas, seed=17, min_groups_per_split=1)
    assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
    for s, p in enumerate(parts):
        idx, _ = _select_exact(rows, p, quotas[s], seed=17+s)
        assert len(idx) == 80
        assert len(set(idx)) == 80
