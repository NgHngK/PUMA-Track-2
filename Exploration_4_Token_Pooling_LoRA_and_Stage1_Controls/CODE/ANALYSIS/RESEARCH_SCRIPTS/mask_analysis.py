from bootstrap import *
from engine import P
from compare import roi_vector
import numpy as np
families=['F_A3','F_GAUSSIAN','F_MASK','F_CLS_MASK','F_SHUFFLE','F_CLS_SHUFFLE'];data={};table=[]
for family in families:
 for seed in [17,29,43]:
  s=json.loads((P/f'RUNS/{family}_s{seed}/summary.json').read_text());m=s['selected'];data[family,seed]=m;table.append({'family':family,'seed':seed,'ROI_F1':m['val']['puma']['fixed10']['macro_f1'],'semantic_F1':m['val']['macro_f1'],'NLL':m['val']['nll'],'ECE15':m['val']['ece15'],'gap':m['gap']})
decisions={}
for family,control in [('F_MASK','F_SHUFFLE'),('F_CLS_MASK','F_CLS_SHUFFLE')]:
 aa=[data[family,s] for s in [17,29,43]];bb=[data['F_A3',s] for s in [17,29,43]];delta=np.array([a['val']['puma']['fixed10']['macro_f1']-b['val']['puma']['fixed10']['macro_f1'] for a,b in zip(aa,bb)]);rd=np.mean([roi_vector(a['val'])-roi_vector(b['val']) for a,b in zip(aa,bb)],0);rng=np.random.default_rng(1701);ci=np.quantile(rd[rng.integers(0,len(rd),(5000,len(rd)))].mean(1),[.025,.975]);rc=np.mean([np.array(a['val']['recall'])-np.array(b['val']['recall']) for a,b in zip(aa,bb)],0);gap=np.mean([a['gap']-b['gap'] for a,b in zip(aa,bb)]);adv={c:float(np.mean([data[family,s]['val']['puma']['fixed10']['macro_f1']-data[c,s]['val']['puma']['fixed10']['macro_f1'] for s in [17,29,43]])) for c in [control,'F_GAUSSIAN']};criteria={'gain':delta.mean()>=.003,'seeds':sum(delta>0)>=2,'interval':ci[0]>0,'tail':rc.min()>=-.10,'gap':gap<=.05,'controls':min(adv.values())>=.002};decisions[family]={'seed_deltas':delta.tolist(),'mean_delta':float(delta.mean()),'ROI_bootstrap95':ci.tolist(),'class_recall_deltas':rc.tolist(),'gap_delta':float(gap),'control_advantage':adv,'criteria':{k:bool(v) for k,v in criteria.items()},'promote':bool(all(criteria.values()))}
csvout(P/'BIOMASK_GUIDED/endpoints.csv',table);dump(P/'BIOMASK_GUIDED/decision.json',decisions);print(json.dumps(decisions,indent=2))
