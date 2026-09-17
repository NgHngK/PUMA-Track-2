from bootstrap import *
from engine import P
from compare import roi_vector
import numpy as np
table=[];deltas=[];rc=[];rd=[]
for seed in [17,29,43]:
 a=json.loads((P/f'RUNS/FINAL_A3_s{seed}/summary.json').read_text())['selected'];b=json.loads((OLD/f'experiments/p3_confirm_A5_s{seed}/summary.json').read_text())['selected']
 for name,m in [('P4_A3',a),('P3_A5',b)]:table.append({'family':name,'seed':seed,'epoch':10,'ROI_F1':m['val']['puma']['fixed10']['macro_f1'],'semantic_F1':m['val']['macro_f1'],'NLL':m['val']['nll'],'ECE15':m['val']['ece15'],'gap':m['gap']})
 deltas.append(a['val']['puma']['fixed10']['macro_f1']-b['val']['puma']['fixed10']['macro_f1']);rc.append(np.array(a['val']['recall'])-np.array(b['val']['recall']));rd.append(roi_vector(a['val'])-roi_vector(b['val']))
rc=np.mean(rc,0);rd=np.mean(rd,0);rng=np.random.default_rng(1701);ci=np.quantile(rd[rng.integers(0,len(rd),(5000,len(rd)))].mean(1),[.025,.975]);passed=np.mean(deltas)>=.003 and sum(d>0 for d in deltas)>=2 and rc.min()>=-.10
csvout(P/'TABLES/confirmation_endpoints.csv',table);dump(P/'TABLES/final_decision.json',{'architecture':'P4_A3' if passed else 'P3_A5','confirmation_pass':bool(passed),'seed_deltas':deltas,'mean_delta':float(np.mean(deltas)),'class_recall_deltas':rc.tolist(),'descriptive_ROI_bootstrap95':ci.tolist(),'scope':'reused development150, fixedepoch10; not independent test','excluded':['multiscale residual','ROI-class sampler','Stage1-aligned training promotion','LoRA','predicted BioMask pooling'],'next_full_scale':'one frozen UNI2-h FOV96 target-neighborhood3x3 representation with retained rank8 TierA interaction head; inverseclass sampling ordinary CE' if passed else 'retained Exploration3 A5; no runner-up shopping'});print('Confirmation',passed,deltas,ci,rc)
