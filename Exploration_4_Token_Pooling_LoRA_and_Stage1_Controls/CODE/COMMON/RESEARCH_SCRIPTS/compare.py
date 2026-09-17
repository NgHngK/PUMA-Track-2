import sys,json,csv,collections
from bootstrap import *
from engine import load_data,extra_metrics,CLASSES,P
import numpy as np,torch
def roi_vector(m):return np.array([sum(r.get('nuclei_'+c,{}).get('f1_score',0) for c in CLASSES)/10 for r in m['puma']['roi_metrics']])
def compare(tag):
 rows,cv,y=load_data();ti=np.flatnonzero(cv>=0);by=collections.defaultdict(list)
 for p in (P/'RUNS').glob('*/summary.json'):
  s=json.loads(p.read_text());c=s['config']
  if c.get('fold',-1)>=0:by[(c['family'],c['seed'])].append((p.parent,s))
 agg={};tables=[];pc=[]
 for (kind,seed),pairs in sorted(by.items()):
  if len(pairs)!=3:continue
  z=np.zeros((len(rows),10),np.float32);seen=[]
  for d,s in pairs:
   pr=np.load(d/f"epoch_{s['config'].get('epochs',10):02}_predictions.npz");z[pr['indices']]=pr['logits'];seen.extend(pr['indices'])
  assert sorted(seen)==ti.tolist();m=extra_metrics([rows[i] for i in ti],y[ti].numpy(),torch.from_numpy(z[ti]));gap=float(np.mean([s['selected']['gap'] for _,s in pairs]));params=pairs[0][1]['trainable_parameters'];agg[(kind,seed)]={'metrics':m,'gap':gap,'params':params,'logits':z[ti]}
  tables.append({'family':kind,'seed':seed,'ROI_F1':m['puma']['fixed10']['macro_f1'],'ROI_P':m['puma']['fixed10']['macro_precision'],'ROI_R':m['puma']['fixed10']['macro_recall'],'pooled_F1':m['puma']['summed']['macro_f1'],'semantic_F1':m['macro_f1'],'accuracy':m['accuracy'],'balanced_accuracy':m['balanced_accuracy'],'NLL':m['nll'],'ECE15':m['ece15'],'parameters':params,'gap':gap,'runtime_seconds':sum(s['runtime_seconds'] for _,s in pairs)})
  for k,c in enumerate(CLASSES):
   pooled=m['puma']['summed']['class_metrics']['nuclei_'+c];pc.append({'family':kind,'seed':seed,'class':c,'support':m['support'][k],'predicted_count':m['predicted'][k],'precision':m['precision'][k],'recall':m['recall'][k],'f1':m['f1'][k],'ROI_F1':m['puma']['fixed10']['f1_by_class']['nuclei_'+c],**pooled})
 out=P/'TABLES'/tag;out.mkdir(exist_ok=True);csvout(out/'combined_cv.csv',tables);csvout(out/'per_class.csv',pc)
 summary=[];prom={};boot={};rng=np.random.default_rng(1701);n=len(roi_vector(agg[('A0',17)]['metrics']));draw=rng.integers(0,n,(5000,n))
 for kind in sorted({k for k,s in agg}):
  aa=[agg[(kind,s)] for s in [17,29,43]];base=[agg[('A0',s)] for s in [17,29,43]];scores=np.array([a['metrics']['puma']['fixed10']['macro_f1'] for a in aa]);delta=scores-np.array([a['metrics']['puma']['fixed10']['macro_f1'] for a in base]);rd=np.mean([roi_vector(a['metrics'])-roi_vector(b['metrics']) for a,b in zip(aa,base)],axis=0);ci=np.quantile(rd[draw].mean(1),[.025,.975]);rc=np.mean([np.array(a['metrics']['recall'])-np.array(b['metrics']['recall']) for a,b in zip(aa,base)],0);gd=float(np.mean([a['gap']-b['gap'] for a,b in zip(aa,base)]))
  control=kind+'_broken' if kind in ['A1','A2','A3','A4'] else ('B_duplicate' if kind in ['B1','B2'] else None)
  advantage=float(scores.mean()-np.mean([agg[(control,s)]['metrics']['puma']['fixed10']['macro_f1'] for s in [17,29,43]])) if control and (control,43) in agg else None
  criteria={'gain':delta.mean()>=.003,'seeds':sum(delta>0)>=2,'interval':ci[0]>0,'tail':rc.min()>=-.10,'gap':gd<=.05,'control':advantage is not None and advantage>=.002}
  eligible=kind in ['A1','A2','A3','A4','B1','B2'];passed=eligible and all(criteria.values());prom[kind]={'pass':bool(passed),'criteria':{k:bool(v) for k,v in criteria.items()},'recall_deltas':rc.tolist(),'gap_delta':gd,'control':control,'control_advantage':advantage}
  summary.append({'family':kind,'ROI_F1_mean':float(scores.mean()),'ROI_F1_sd':float(scores.std(ddof=1)),'delta_vs_A5':float(delta.mean()),'positive_seeds':int(sum(delta>0)),'bootstrap_low':float(ci[0]),'bootstrap_high':float(ci[1]),'control_advantage':advantage,'parameters':aa[0]['params'],'promoted':bool(passed)})
  boot[kind]={'ROI_deltas':rd.tolist(),'CI':ci.tolist(),'seed_deltas':delta.tolist(),'ROI_order':sorted({rows[i]['roi'] for i in ti})}
 csvout(out/'seed_summary.csv',summary);dump(out/'promotion.json',prom);dump(out/'uncertainty.json',boot);dump(out/'oof_metrics.json',{f'{k}_{s}':{kk:vv for kk,vv in a.items() if kk!='logits'} for (k,s),a in agg.items()});np.savez_compressed(out/'oof_logits.npz',**{f'{k}_{s}':a['logits'] for (k,s),a in agg.items()})
 passing=sorted([r for r in summary if r['family'].startswith('A') and r['promoted']],key=lambda r:-r['ROI_F1_mean']);chosen='A0'
 if passing:chosen=min([r for r in passing if passing[0]['ROI_F1_mean']-r['ROI_F1_mean']<=.002],key=lambda r:r['parameters'])['family']
 dump(out/'selection.json',{'best_target':chosen,'scope':'development CV; confirmation still required','B3_justified':prom.get('B1',{}).get('pass',False) and prom.get('B2',{}).get('pass',False)})
 print(json.dumps({'best_target':chosen,'summary':summary},indent=2));return chosen
if __name__=='__main__':compare(sys.argv[1])
