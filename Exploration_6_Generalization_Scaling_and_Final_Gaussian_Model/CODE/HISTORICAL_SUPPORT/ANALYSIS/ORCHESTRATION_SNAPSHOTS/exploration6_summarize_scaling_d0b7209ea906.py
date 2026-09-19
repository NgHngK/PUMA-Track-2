from pathlib import Path
import json,csv
import numpy as np
R=Path(r'C:\Users\Hngk\Documents\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
CL=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
def write(name,rows):
 with (O/'TABLES'/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
rows=[];summaries={}
for n in [300,600,900]:
 for seed in [17,29,43]:
  p=O/f'EXPERIMENTS/p6_A_D{n}_s{seed}/summary.json'
  if not p.exists():raise RuntimeError(f'Wait for all preregistered controls: {p}')
  d=json.loads(p.read_text());s=d['selected'];summaries[n,seed]=d
  assert d['selection_mode']=='fixed_final' and d['selected_epoch']==10
  rows.append(dict(run_id=d['experiment_id'],dataset=n,seed=seed,train_f1=s['train']['macro_f1'],dev_f1=s['dev']['macro_f1'],roi_f1=s['dev']['puma']['fixed10']['macro_f1'],train_loss=s['train']['nll'],dev_loss=s['dev']['nll'],ece15=s['dev']['ece15'],gap=s['generalization_gap_macro_f1'],unique_epoch10=s['unique_examples'],unique_groups_epoch10=s['unique_groups'],repeat_epoch10=s['repeat_fraction'],parameters=d['trainable_parameters'],epochs=d['epochs_run']))
write('DATA_SCALING.csv',rows)
agg=[]
for n in [300,600,900]:
 r=[x for x in rows if x['dataset']==n]; a=dict(dataset=n)
 for k in ['train_f1','dev_f1','roi_f1','train_loss','dev_loss','gap','ece15']:
  v=[x[k] for x in r];a[k+'_mean']=float(np.mean(v));a[k+'_sd']=float(np.std(v,ddof=1))
 agg.append(a)
write('DATA_SCALING_AGGREGATE.csv',agg)
comparisons=[]
for n in [600,900]:
 deltas=[];classes=[]
 for seed in [17,29,43]:
  c=summaries[n,seed]['selected']['dev'];b=summaries[300,seed]['selected']['dev']
  def rois(m):return np.array([sum(r.get('nuclei_'+name,{}).get('f1_score',0.0) for name in CL)/10 for r in m['puma']['roi_metrics']])
  bc,cc=rois(b),rois(c)
  assert np.isclose(bc.mean(),b['puma']['fixed10']['macro_f1']) and np.isclose(cc.mean(),c['puma']['fixed10']['macro_f1'])
  deltas.append(cc-bc);classes.append(np.array(c['recall'])-np.array(b['recall']))
 effects=np.mean(deltas,axis=0);rng=np.random.default_rng(1706);boot=effects[rng.integers(0,len(effects),size=(2000,len(effects)))].mean(1)
 classdelta=np.mean(classes,axis=0)
 comparisons.append(dict(candidate=f'D{n}',control='D300',paired_roi_delta=float(effects.mean()),roi_ci95=np.quantile(boot,[.025,.975]).tolist(),positive_seeds=int(sum(np.mean(x)>0 for x in deltas)),mean_recall_delta=dict(zip(CL,classdelta.tolist())),class_harm=[CL[k] for k in range(10) if classdelta[k]<-.10],uncertainty_unit='56 ROI groups; averaged across the three fixed seeds before ROI bootstrap',estimand='matched DEV conditional subset V17 fixed10 ROI-F1, not complete-proposal evaluation'))
out=dict(aggregate=agg,comparisons=comparisons,next_stage='B raw AUG0 parity, AUG1 geometry, AUG2 geometry+fixed H/E OD',holdouts_opened=False)
(O/'TABLES/DATA_SCALING_SUMMARY.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
write('EXPERIMENT_REGISTRY.csv',[dict(experiment_group='A',run_id=r['run_id'],status='COMPLETE',dataset=r['dataset'],seed=r['seed'],selected_epoch=10,selected_roi_f1=r['roi_f1'],summary=str(O/f"EXPERIMENTS/{r['run_id']}/summary.json")) for r in rows])
print(json.dumps(out,separators=(',',':')))
