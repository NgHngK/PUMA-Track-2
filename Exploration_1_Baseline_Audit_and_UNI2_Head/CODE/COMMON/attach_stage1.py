"""Attach frozen OOF centers to GT labels with one-to-one geometry-only matching.

For semantic diagnostics only. All unmatched proposals must remain in end-to-end
evaluation. Supply caller-verified OOF provenance; this script never runs Stage 1.
"""
import argparse,csv,json
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from dataset import read_manifest,CLASSES

def main():
    p=argparse.ArgumentParser();p.add_argument('--gt-manifest',required=True);p.add_argument('--proposals',required=True)
    p.add_argument('--out',required=True);p.add_argument('--radius',type=float,required=True)
    p.add_argument('--provenance',required=True,help='JSON: checkpoint hashes, coordinate units, and excluded outer groups')
    a=p.parse_args();gt=read_manifest(a.gt_manifest)
    if a.radius<=0:raise ValueError('Positive radius in ROI pixels required')
    provenance=json.loads(Path(a.provenance).read_text())
    held={r['group'] for r in gt if r['split']!='train'}
    if provenance.get('coordinate_units')!='roi_level0_pixels' or not provenance.get('checkpoint_sha256'):
        raise ValueError('Missing Stage-1 checkpoint/coordinate provenance')
    if not held.issubset(set(provenance.get('excluded_outer_groups',[]))):
        raise ValueError('Cannot verify Stage-1 exclusion of outer held-out groups')
    if not provenance.get('training_centers_are_oof',False):raise ValueError('Training centers must be OOF')
    with open(a.proposals,newline='',encoding='utf-8') as f:pr=list(csv.DictReader(f))
    if not pr or not {'uid','roi','x','y'}.issubset(pr[0]):raise ValueError('Invalid proposals CSV')
    if len({r['uid'] for r in pr})!=len(pr):raise ValueError('Duplicate proposal uid')
    gp={};pp={}
    for r in gt:gp.setdefault(r['roi'],[]).append(r)
    for r in pr:
        if r['roi'] not in gp:raise ValueError('Unknown ROI '+r['roi'])
        pp.setdefault(r['roi'],[]).append(r)
    output=[];audit=[]
    for roi,gg in gp.items():
        rr=pp.get(roi,[])
        if not rr:audit.append(dict(roi=roi,gt=len(gg),proposals=0,matched=0));continue
        xy=np.array([[float(r['x']),float(r['y'])] for r in rr])
        if not np.isfinite(xy).all():raise ValueError('Invalid proposal coordinates')
        dist=cdist(np.array([[r['x'],r['y']] for r in gg]),xy)
        # Large cardinality penalty ensures maximum valid matches, then minimum distance.
        penalty=(len(gg)+1)*(a.radius+1)
        cost=np.concatenate([np.where(dist<=a.radius,dist,2*penalty),np.full((len(gg),len(gg)),penalty)],axis=1)
        gi,pi=linear_sum_assignment(cost);matched=0
        for g,i in zip(gi,pi):
            if i>=len(rr) or dist[g,i]>a.radius:continue
            r=gg[g].copy();r.update(uid=rr[i]['uid'],x=float(rr[i]['x']),y=float(rr[i]['y']),coordinate_source='stage1_oof')
            output.append(r);matched+=1
        audit.append(dict(roi=roi,gt=len(gg),proposals=len(rr),matched=matched))
    if not output:raise ValueError('No matched nuclei')
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    out.with_suffix('.json').write_text(json.dumps({'classes':CLASSES,'source':'matched frozen Stage-1 OOF centers'},indent=2))
    out.with_suffix('.matching.json').write_text(json.dumps({'radius':a.radius,'audit':audit,'provenance':provenance},indent=2))

if __name__=='__main__':main()
