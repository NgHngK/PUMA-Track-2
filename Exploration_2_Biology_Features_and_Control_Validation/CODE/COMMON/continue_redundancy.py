import sys,json,time
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from dataset import read_manifest
def main():
    rows=read_manifest(R/'01_sample_definition/sample_manifest.csv');tr=np.array([r['split']=='train' for r in rows])
    b=np.load(R/'02_cache/biology/tierA.npy');names=json.loads((R/'02_cache/biology/tierA_schema.json').read_text())['names']
    x=np.load(R/'02_cache/uni2_features/gt_fov96/features.npy').astype(float);x=(x-x.mean(1,keepdims=True))/np.sqrt(x.var(1,keepdims=True)+1e-5)
    bm=b[tr].mean(0);bs=np.maximum(b[tr].std(0),1e-6);b=(b-bm)/bs;xm=x[tr].mean(0);x=x-xm
    with threadpool_limits(limits=2):
        # Prespecified alpha1, training-only fit; no validation alpha search.
        coefficients=np.linalg.solve(x[tr]@x[tr].T+np.eye(tr.sum()),b[tr]);pred=(x[~tr]@x[tr].T)@coefficients
        den=((b[~tr]-b[~tr].mean(0))**2).sum(0);r2=np.divide(((b[~tr]-pred)**2).sum(0),den,out=np.full(len(names),np.nan),where=den>1e-12);r2=1-r2
        corr=np.corrcoef(b[tr],rowvar=False);singular=np.linalg.svd(b[tr]-b[tr].mean(0),compute_uv=False);v=singular**2;v=v/v.sum();rank=float(np.exp(-(v[v>0]*np.log(v[v>0])).sum()))
    out=R/'90_comparative_analysis';out.mkdir(exist_ok=True)
    report={'label':'exploratory reconstruction, not a classifier or causal redundancy proof','fit':'dual ridge alpha1; train-centered LN UNI2; train-standardized TierA','FOV':96,'train_n':int(tr.sum()),'validation_n':int((~tr).sum()),'held_out_R2_by_feature':dict(zip(names,[None if not np.isfinite(z) else float(z) for z in r2])),'train_biology_effective_rank':rank,'train_correlation':corr.tolist()}
    (out/'biology_redundancy.json').write_text(json.dumps(report,indent=2));print('Biology effective rank',rank,'median heldout R2',float(np.nanmedian(r2)))
if __name__=='__main__':main()
