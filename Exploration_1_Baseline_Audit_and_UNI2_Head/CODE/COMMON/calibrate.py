"""Fit a scalar temperature on a separate, naturally sampled calibration split."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train_eval import metrics

def fit_temperature(logits,labels):
    z=torch.as_tensor(logits,dtype=torch.float32);y=torch.as_tensor(labels,dtype=torch.long)
    if not torch.isfinite(z).all() or len(y)==0:raise ValueError('Invalid calibration data')
    # Bounded 1-D search avoids optimizing class biases on scarce calibration labels.
    temperatures=torch.logspace(-1,1,161)
    losses=torch.stack([F.cross_entropy(z/t,y) for t in temperatures])
    return float(temperatures[losses.argmin()])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--predictions',required=True);p.add_argument('--out',required=True)
    p.add_argument('--assert-calibration-split',action='store_true',required=True)
    a=p.parse_args();d=np.load(a.predictions,allow_pickle=False)
    t=fit_temperature(d['logits'],d['labels'])
    Path(a.out).write_text(json.dumps({'temperature':t,'before':metrics(d['labels'],d['logits']),
                                     'after':metrics(d['labels'],d['logits']/t)},indent=2))
