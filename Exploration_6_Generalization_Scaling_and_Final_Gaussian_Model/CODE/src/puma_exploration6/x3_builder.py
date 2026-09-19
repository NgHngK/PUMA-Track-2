from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import numpy as np

from .constants import CLASSES,NUM_CLASSES
from .manifest import write_manifest

# Exact 1x historical Exploration-3 research-cohort class support. Exploration-6 x3
# preserves that lineage rather than silently changing to a class-balanced cohort.
BASE_TRAIN_QUOTA=[71,37]+[24]*8
BASE_VAL_QUOTA=[39,15]+[12]*8
TRAIN_QUOTA=[3*x for x in BASE_TRAIN_QUOTA]              # 900
# Three historical validation equivalents total 450; split once into fresh DEV/LOCKED.
DEV_QUOTA=[58,23]+[18]*8                                  # 225
LOCKED_QUOTA=[59,22]+[18]*8                               # 225


def _group_stats(rows):
    stats=defaultdict(lambda:np.zeros(NUM_CLASSES,dtype=int))
    for r in rows:stats[str(r["group"])][int(r["label"])]+=1
    return stats


def _candidate_group_split(rows,seed: int,trials: int=1000):
    """Search group-disjoint 2/3,1/6,1/6 partitions with enough class support.

    Candidate generation operates on group IDs, never individual nuclei, so leakage
    is impossible by construction. Objective prioritizes minimum class-group support.
    """
    stats=_group_stats(rows);groups=np.array(sorted(stats),dtype=object);rng=np.random.default_rng(seed);best=None
    req=[np.array(TRAIN_QUOTA),np.array(DEV_QUOTA),np.array(LOCKED_QUOTA)]
    for _ in range(trials):
        perm=groups.copy();rng.shuffle(perm);n=len(perm);a=max(1,round(n*2/3));b=max(a+1,round(n*5/6))
        parts=[set(perm[:a]),set(perm[a:b]),set(perm[b:])]
        if not parts[1] or not parts[2]:continue
        counts=[];group_support=[];feasible=True
        for pi,p in enumerate(parts):
            c=sum((stats[g] for g in p),np.zeros(NUM_CLASSES,dtype=int));counts.append(c)
            gs=np.array([sum(stats[g][k]>0 for g in p) for k in range(NUM_CLASSES)])
            group_support.append(gs)
            if (c<req[pi]).any():feasible=False;break
        if not feasible:continue
        min_groups=min(int(gs.min()) for gs in group_support)
        # Secondary objective: balance group counts and avoid concentration.
        score=(min_groups,float(sum(np.log1p(gs).sum() for gs in group_support)))
        if best is None or score>best[0]:best=(score,parts,counts,group_support)
    if best is None:raise ValueError("unable to find a group-disjoint split satisfying x3 per-class quotas; increase trials or provide stronger grouping metadata")
    return best[1],best[2],best[3]


def _diversity_order(rows,class_id: int,allowed_groups: set[str],seed: int) -> list[int]:
    by=defaultdict(list)
    for i,r in enumerate(rows):
        if int(r["label"])==class_id and str(r["group"]) in allowed_groups:by[str(r["group"])].append(i)
    if not by:raise ValueError(f"no candidates for class {CLASSES[class_id]}")
    rng=np.random.default_rng(seed+class_id*1009);groups=sorted(by)
    for g in groups:rng.shuffle(by[g])
    rng.shuffle(groups);out=[];depth=0
    while True:
        added=False
        for g in groups:
            if depth<len(by[g]):out.append(by[g][depth]);added=True
        if not added:break
        depth+=1
    return out


def _select(rows,groups: set[str],quotas: list[int],seed: int):
    selected=[];orders={}
    for c,q in enumerate(quotas):
        order=_diversity_order(rows,c,groups,seed);orders[c]=order
        if len(order)<q:raise ValueError(f"insufficient class {CLASSES[c]}: need {q}, have {len(order)}")
        selected.extend(order[:q])
    if len(selected)!=sum(quotas) or len(set(selected))!=len(selected):raise AssertionError("selection size/uniqueness failure")
    return selected,orders


def _clone(row,split):
    x=dict(row);x["split"]=split;return x


def build_x3(rows: list[dict],out_dir: str | Path,seed: int=17,trials: int=1000) -> dict:
    if len({r["uid"] for r in rows})!=len(rows):raise ValueError("source rows contain duplicate uids")
    out_dir=Path(out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"x3 output directory is not empty; preserve the prior dataset and choose a new directory: {out_dir}")
    # The x3 cohort must be drawn from a fresh unpartitioned annotation pool.
    # Refuse manifests that already contain DEV/LOCKED/TEST assignments so prior
    # development data cannot be silently recycled into a supposedly fresh split.
    source_splits={str(r.get("split", "")).strip().lower() for r in rows}
    if source_splits != {"train"}:
        raise ValueError(f"x3 source must be an unpartitioned all-train pool; found splits={sorted(source_splits)}")
    parts,available,group_support=_candidate_group_split(rows,seed,trials);train_groups,dev_groups,locked_groups=parts
    train_idx,orders=_select(rows,train_groups,TRAIN_QUOTA,seed)
    dev_idx,_=_select(rows,dev_groups,DEV_QUOTA,seed+1);locked_idx,_=_select(rows,locked_groups,LOCKED_QUOTA,seed+2)
    all_idx=train_idx+dev_idx+locked_idx
    if len(set(all_idx))!=1350:raise AssertionError("x3 selected nuclei are not unique")
    if (set(train_groups)&set(dev_groups)) or (set(train_groups)&set(locked_groups)) or (set(dev_groups)&set(locked_groups)):raise AssertionError("group leakage")
    out_dir.mkdir(parents=True,exist_ok=True)
    # Nested training subsets exactly preserve the historical Exploration-3 class support
    # at 1x, 2x and 3x while increasing unique/group-diverse nuclei.
    nested={}
    for multiplier,name in [(1,"D300"),(2,"D600"),(3,"D900")]:
        tr=[]
        quotas=[multiplier*q for q in BASE_TRAIN_QUOTA]
        for c,q in enumerate(quotas):tr.extend(orders[c][:q])
        manifest=[_clone(rows[i],"train") for i in tr]+[_clone(rows[i],"dev") for i in dev_idx]+[_clone(rows[i],"locked") for i in locked_idx]
        path=out_dir/f"{name}.csv";write_manifest(path,manifest,{"dataset":name,"seed":seed,"group_disjoint":True,"locked_unseen":True});nested[name]=str(path)
    audit=[]
    for split,idx,gs in [("train",train_idx,train_groups),("dev",dev_idx,dev_groups),("locked",locked_idx,locked_groups)]:
        for c in range(NUM_CLASSES):
            ci=[i for i in idx if int(rows[i]["label"])==c];cg={str(rows[i]["group"]) for i in ci};counts=defaultdict(int)
            for i in ci:counts[str(rows[i]["group"])]+=1
            max_share=max(counts.values())/len(ci) if ci else 0
            audit.append({"split":split,"class_id":c,"class_name":CLASSES[c],"nuclei":len(ci),"groups":len(cg),"max_group_share":max_share})
    with (out_dir/"group_support.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)
    summary={"selected_total":1350,"train":900,"dev":225,"locked":225,"seed":seed,"manifests":nested,
             "train_class_quota":TRAIN_QUOTA,"dev_class_quota":DEV_QUOTA,"locked_class_quota":LOCKED_QUOTA,
             "historical_train_1x_quota":BASE_TRAIN_QUOTA,"historical_val_1x_quota":BASE_VAL_QUOTA,
             "partition_group_counts":[len(p) for p in parts],"minimum_available_group_support":[int(x.min()) for x in group_support]}
    (out_dir/"summary.json").write_text(__import__("json").dumps(summary,indent=2),encoding="utf-8");return summary
