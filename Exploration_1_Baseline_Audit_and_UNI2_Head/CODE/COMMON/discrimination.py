import sys,json
from pathlib import Path
ROOT=Path(__file__).parent;sys.path.insert(0,str(ROOT.parent/'outputs'))
import numpy as np
from sklearn.metrics import average_precision_score
from dataset import read_manifest,CLASSES
from train_eval import metrics
import torch

rows=read_manifest(ROOT/'uni2_sample.csv');x=np.load(ROOT/'uni2_features.npy');tr=np.array([r['split']=='train' for r in rows]);y=np.array([r['label'] for r in rows])
x=x/np.linalg.norm(x,axis=1,keepdims=True);h=x[tr];v=x[~tr];yt=y[tr];yv=y[~tr]
centroids=np.stack([h[yt==c].mean(0) for c in range(10)]);centroids/=np.linalg.norm(centroids,axis=1,keepdims=True)
centroid_logits=v@centroids.T
results={'centroid':metrics(yv,centroid_logits)}
similarity=v@h.T
for k in [1,5]:
    ix=np.argsort(-similarity,axis=1)[:,:k]
    votes=np.stack([(yt[ix]==c).sum(1) for c in range(10)],axis=1).astype(float)
    # Use the lower class ID for exact ties so the result is reproducible.
    results['knn'+str(k)]=metrics(yv,votes)
for name in ['ce','la','balanced_ce']:
    d=np.load(ROOT/('uni2_results/'+name+'_best.npz'));probs=torch.from_numpy(d['logits']).softmax(-1).numpy()
    results[name+'_AP']=[float(average_precision_score(yv==c,probs[:,c])) for c in range(10)]
results['caution']='exploratory; centroid/knn logits are similarity/vote scores, so their NLL/ECE are not probability-calibration claims; deterministic lowest-index tie breaking'
(ROOT/'discrimination.json').write_text(json.dumps(results,indent=2))
print({k:{'f1':v['macro_f1'],'apoptosis_recall':v['recall'][9]} for k,v in results.items() if isinstance(v,dict)})
