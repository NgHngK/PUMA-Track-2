from bootstrap import *
from engine import load_data,extra_metrics,P
from compare import roi_vector
import numpy as np,torch
rows,cv,y=load_data();ti=np.flatnonzero(cv>=0);table=[];metrics={}
for kind in ['CLS','A3']:
 for seed in [17,29,43]:
  z=np.zeros((len(rows),10),np.float32)
  for fold in range(3):
   f=OLD/f'experiments/p3_A0_real_s{seed}_cv{fold}/predictions.npz' if kind=='CLS' else P/f'RUNS/DIAG_A3LINEAR_s{seed}_cv{fold}/epoch_10_predictions.npz'
   d=np.load(f);ids=np.flatnonzero(cv==fold) if kind=='CLS' else d['indices'];z[ids]=d['logits']
  m=extra_metrics([rows[i] for i in ti],y[ti].numpy(),torch.from_numpy(z[ti]));metrics[kind,seed]=m
  table.append({'representation':kind,'seed':seed,'ROI_F1':m['puma']['fixed10']['macro_f1'],'semantic_F1':m['macro_f1'],'NLL':m['nll'],'ECE15':m['ece15'],'parameters':15370})
delta=[metrics['A3',s]['puma']['fixed10']['macro_f1']-metrics['CLS',s]['puma']['fixed10']['macro_f1'] for s in [17,29,43]];rd=np.mean([roi_vector(metrics['A3',s])-roi_vector(metrics['CLS',s]) for s in [17,29,43]],0);rng=np.random.default_rng(1701);ci=np.quantile(rd[rng.integers(0,len(rd),(5000,len(rd)))].mean(1),[.025,.975])
D=P/'TARGET_POOLING/DIAGNOSTICS';csvout(D/'plain_head_endpoints.csv',table);dump(D/'plain_head_comparison.json',{'seed_deltas':delta,'mean_delta':float(np.mean(delta)),'ROI_bootstrap95':ci.tolist(),'selection_eligible':False,'interpretation':'Same linear head without TierA; diagnostic separates representation gain from interaction-head gain. Reuses historical CLS predictions.'});print(table,delta,ci)
