from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(r'C:\Users\Hngk\Documents\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
CL=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
S=[17,29,43];arms=['inverse','natural','tempered','crt_reset','crt_noreset'];m={};rows=[]
def roi(x):
 r=x['puma']['roi_metrics'];v=np.array([sum(a.get('nuclei_'+c,{}).get('f1_score',0) for c in CL)/10 for a in r])
 assert np.isclose(v.mean(),x['puma']['fixed10']['macro_f1'])
 return v
for arm in arms:
 for seed in S:
  run=f'p6_A_D900_s{seed}' if arm=='inverse' else f'p6_C_cached_{arm}_s{seed}'
  p=O/f'EXPERIMENTS/{run}/summary.json';d=json.loads(p.read_text());h=(d['decoupled']['epochs'][-1] if d.get('decoupled') else d['selected']);m[arm,seed]=h
  assert np.isclose(h['dev']['puma']['fixed10']['macro_f1'],d['final_model_score'])
  rows.append(dict(arm=arm,seed=seed,run_id=run,train_f1=h['train']['macro_f1'],dev_f1=h['dev']['macro_f1'],roi_f1=d['final_model_score'],gap=h['train']['macro_f1']-h['dev']['macro_f1'],dev_nll=h['dev']['nll'],ece15=h['dev']['ece15'],endpoint='decoupled_epoch5' if d.get('decoupled') else 'joint_epoch10',runtime_seconds=d['runtime_seconds'],checkpoint=d['final_model_checkpoint'],summary_path=str(p),summary_sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
comparisons=[]
for arm,control in [(a,'inverse') for a in arms[1:]]+[('crt_reset','crt_noreset')]:
 delta=np.array([roi(m[arm,s]['dev'])-roi(m[control,s]['dev']) for s in S]);effect=delta.mean(0);rng=np.random.default_rng(1706);boot=effect[rng.integers(0,len(effect),(2000,len(effect)))].mean(1);ci=np.quantile(boot,[.025,.975]);cd=np.mean([np.array(m[arm,s]['dev']['recall'])-np.array(m[control,s]['dev']['recall']) for s in S],0)
 gap=np.mean([m[arm,s]['generalization_gap_macro_f1']-m[control,s]['generalization_gap_macro_f1'] for s in S]);harm=[CL[i] for i in range(10) if cd[i]<-.10];positive=int((delta.mean(1)>0).sum())
 comparisons.append(dict(candidate=arm,control=control,roi_delta=float(effect.mean()),ci95=ci.tolist(),positive_seeds=positive,mean_recall_delta=dict(zip(CL,cd.tolist())),class_harm=harm,gap_change=float(gap),passes_numeric_guards=bool(effect.mean()>=.003 and ci[0]>0 and positive>=2 and not harm and gap<=.05),promotion='NOT AUTHORIZED: conditional on B augmentation resolution',bootstrap='2000 resamples of56pairedROI effects averaged across3seeds; seed1706'))
with (O/'TABLES/CACHED_LONG_TAIL_CONTROLS.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
out=dict(status='COMPLETE_CONDITIONAL_CACHED_CONTROLS',holdouts_opened=False,means={a:{k:float(np.mean([r[k] for r in rows if r['arm']==a])) for k in ['train_f1','dev_f1','roi_f1','gap','dev_nll','ece15']} for a in arms},comparisons=comparisons)
(O/'TABLES/CACHED_LONG_TAIL_SUMMARY.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out))
