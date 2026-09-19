"""Run only when all nine frozen raw-augmentation trials finish."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(r'C:\Users\Hngk\Documents\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
CL=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis'];S=[17,29,43];data={};rows=[];parity=[]
for arm in ['AUG0','AUG1','AUG2']:
 for seed in S:
  root=O/f'EXPERIMENTS/p6_B_{arm}_s{seed}';p=root/'summary.json'
  if not p.exists():raise RuntimeError(f'All nine runs must complete before interpretation; missing {p}')
  d=json.loads(p.read_text());assert d['selection_mode']=='fixed_final' and d['selected_epoch']==10;h=d['selected'];data[arm,seed]=h
  rows.append(dict(arm=arm,seed=seed,run_id=root.name,train_f1=h['train']['macro_f1'],dev_f1=h['dev']['macro_f1'],roi_f1=h['dev']['puma']['fixed10']['macro_f1'],gap=h['generalization_gap_macro_f1'],dev_nll=h['dev']['nll'],ece15=h['dev']['ece15'],summary=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
  if arm=='AUG0':
   a=np.load(O/f'EXPERIMENTS/p6_A_D900_s{seed}/predictions/final_dev.npz');b=np.load(root/'predictions/final_dev.npz');assert np.array_equal(a['uids'],b['uids']) and np.array_equal(a['labels'],b['labels']);delta=abs(a['logits']-b['logits'])
   parity.append(dict(seed=seed,max_abs_logit_error=float(delta.max()),mean_abs_logit_error=float(delta.mean()),prediction_disagreements=int((a['logits'].argmax(1)!=b['logits'].argmax(1)).sum()),scope='Raw physical1 accumulation64 versus cached batch64; identical ordered DEV UIDs; diagnostic measured parity, no exact-float claim'))
def roi(m):
 return np.array([sum(r.get('nuclei_'+c,{}).get('f1_score',0) for c in CL)/10 for r in m['puma']['roi_metrics']])
comparisons=[]
for arm,control in [('AUG1','AUG0'),('AUG2','AUG0'),('AUG2','AUG1')]:
 delta=np.array([roi(data[arm,s]['dev'])-roi(data[control,s]['dev']) for s in S]);effects=delta.mean(0);rng=np.random.default_rng(1706);boot=effects[rng.integers(0,len(effects),(2000,len(effects)))].mean(1);ci=np.quantile(boot,[.025,.975]);cd=np.mean([np.array(data[arm,s]['dev']['recall'])-np.array(data[control,s]['dev']['recall']) for s in S],axis=0);gap=float(np.mean([data[arm,s]['generalization_gap_macro_f1']-data[control,s]['generalization_gap_macro_f1'] for s in S]));positive=int((delta.mean(1)>0).sum());harm=[CL[i] for i in range(10) if cd[i]<-.1]
 comparisons.append(dict(candidate=arm,control=control,roi_delta=float(effects.mean()),roi_ci95=ci.tolist(),positive_seeds=positive,class_recall_delta=dict(zip(CL,cd.tolist())),class_harm=harm,gap_change=gap,passes_predeclared_numeric_guards=bool(effects.mean()>=.003 and ci[0]>0 and positive>=2 and not harm and gap<=.05)))
with (O/'TABLES/AUGMENTATION_ENDPOINTS.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(status='NINE_RUNS_COMPLETE_REQUIRES_SCIENTIFIC_REVIEW_BEFORE_NEXT_STAGE',parity=parity,comparisons=comparisons,means={arm:{key:float(np.mean([r[key] for r in rows if r['arm']==arm])) for key in ['train_f1','dev_f1','roi_f1','gap','dev_nll','ece15']} for arm in ['AUG0','AUG1','AUG2']},holdouts_opened=False)
(O/'TABLES/AUGMENTATION_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary))
