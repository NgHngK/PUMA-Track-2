from pathlib import Path
import json,csv,hashlib,sys,collections
import numpy as np
import torch
from torch.utils.data import DataLoader
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';T=O/'TABLES'
sys.path.insert(0,str(C/'src'))
from puma_exploration6.config import ExperimentConfig
from puma_exploration6.manifest import read_manifest
from puma_exploration6.sampling import build_sampler
CL=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis'];M='NOT RECORDED BY EXECUTED TRAINER'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(x):return json.dumps(x,separators=(',',':')) if isinstance(x,(dict,list)) else x
def wr(name,fields):
 f=(T/name).open('w',newline='',encoding='utf-8');w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();return f,w
b=['prompt','experiment_group','scenario_id','run_id','seed','fold','phase','epoch','split']
fg,wg=wr('EPOCH_GLOBAL_METRICS.csv',b+['objective','nll','macro_precision','macro_recall','macro_f1','balanced_accuracy_fixed10','accuracy','ece15','roi_macro_precision','roi_macro_recall','roi_macro_f1','pooled_macro_f1','lr','grad_norm_mean','draws','unique_nuclei','unique_groups','repeat_fraction','best_so_far','minimum_delta_pass','patience_counter','selected','checkpoint_saved','source_sha256'])
fp,wp=wr('EPOCH_PER_CLASS_METRICS.csv',b+['class_name','support','predicted','precision','recall','f1','TP','FP','FN','dominant_confusion'])
fc,wc=wr('CONFUSION_BY_EPOCH.csv',b+['true_class','predicted_class','count'])
fe,we=wr('EPOCH_SAMPLER_EXPOSURE.csv',b+['class_name','available_unique','draws','unique_nuclei','repeated_draws','mean_exposure','maximum_exposure','unique_groups','provenance'])
fd,wd=wr('EPOCH_MODEL_DIAGNOSTICS.csv',b+['parameter_norms','train_diagnostics','dev_diagnostics'])
fr,wrn=wr('RUN_CONFIG_SNAPSHOTS.csv',['run_id','seed','config_sha256','config','provenance','checkpoint_rule','selected_epoch','final_executed_epoch'])
fs,ws=wr('TRAINING_SCENARIOS.csv',['run_id','experiment_group','hypothesis','changed','fixed','train_nuclei','dev_nuclei','locked_nuclei','train_groups','dev_groups','locked_groups','grouping','manifest_sha256','status'])
summary=[];epochs_total=0
for root in sorted((O/'EXPERIMENTS').iterdir()):
 hp=root/'history.jsonl';cp=root/'config.json'
 if not hp.is_file() or not cp.is_file():continue
 cfg=ExperimentConfig.load(cp);run=cfg.experiment_id;group=run.split('_')[1];data=read_manifest(cfg.manifest,require_files=False);train=[r for r in data if r['split']=='train'];labels=[r['label'] for r in train]
 history=[json.loads(line) for line in hp.read_text().splitlines() if line.strip()];sp=root/'summary.json';result=json.loads(sp.read_text()) if sp.exists() else {};prov=json.loads((root/'provenance.json').read_text());hhash=sha(hp)
 sampler,_,_=build_sampler(torch.tensor(labels),cfg.sampler,cfg.seed)
 # Replay in a separate process with the exact local sampler generator. Verify
 # every logged exposure/coverage statistic before using supplementary counts.
 loader=DataLoader(list(range(len(train))),batch_size=cfg.optimizer.batch_size,shuffle=sampler is None,sampler=sampler,generator=torch.Generator().manual_seed(cfg.seed),num_workers=0)
 best=-float('inf');stale=0
 decoupled=(result.get('decoupled') or {}).get('epochs',[])
 all_epochs=[('joint',h) for h in history]+[('decoupled',h) for h in decoupled]
 phase_previous='joint'
 for phase,h0 in all_epochs:
  h=dict(h0)
  if phase!=phase_previous:
   sampler,_,_=build_sampler(torch.tensor(labels),cfg.decoupled.sampler,cfg.seed+7000)
   loader=DataLoader(list(range(len(train))),batch_size=cfg.optimizer.batch_size,shuffle=sampler is None,sampler=sampler,generator=torch.Generator().manual_seed(cfg.seed+7000),num_workers=0)
   best=-float('inf');stale=0;phase_previous=phase
  if phase=='decoupled':
   h['selection_score']=h['dev']['puma']['fixed10']['macro_f1'];h['lr']=[cfg.decoupled.retrain_lr]
  epochs_total+=1;draw=[int(i) for batch in loader for i in batch.tolist()];freq=collections.Counter(draw);exp=[sum(n for i,n in freq.items() if labels[i]==k) for k in range(10)];unq=[sum(labels[i]==k for i in freq) for k in range(10)]
  assert exp==h['exposure'] and unq==h['unique_examples_by_class'] and len(freq)==h['unique_examples']
  assert len({train[i]['group'] for i in freq})==h['unique_groups']
  score=h['selection_score'];improved=score>best+cfg.optimizer.min_delta
  if improved:best=score;stale=0
  else:stale+=1
  base=dict(prompt=6,experiment_group=group,scenario_id='_'.join(run.split('_')[:-1]),run_id=run,seed=cfg.seed,fold='NOT APPLICABLE',phase=phase,epoch=h['epoch'])
  for split in ['train','dev']:
   m=h[split];p=m['puma'];g={**base,'split':split,**{k:m.get(k,M) for k in ['nll','macro_precision','macro_recall','macro_f1','balanced_accuracy_fixed10','accuracy','ece15']}}
   g.update(objective=h['objective'] if split=='train' else 'NOT APPLICABLE',lr=js(h['lr']),grad_norm_mean=h['grad_norm_mean'],draws=len(draw),unique_nuclei=len(freq),unique_groups=h['unique_groups'],repeat_fraction=h['repeat_fraction'],best_so_far=best,minimum_delta_pass=improved,patience_counter=stale,selected=h['epoch']==result.get('selected_epoch'),checkpoint_saved=improved or h['epoch']==result.get('epochs_run'),source_sha256=hhash)
   if phase=='decoupled':
    g.update(selected=h['epoch']==cfg.decoupled.retrain_epochs,checkpoint_saved=h['epoch']==cfg.decoupled.retrain_epochs,source_sha256=sha(sp),minimum_delta_pass='NOT APPLICABLE: fixed-final cRT',patience_counter='NOT APPLICABLE: fixed-final cRT')
   g.update({f'roi_{k}':p['fixed10'][k] for k in ['macro_precision','macro_recall','macro_f1']});g['pooled_macro_f1']=p['summed']['macro_f1'];wg.writerow(g)
   cm=np.array(m['confusion'])
   for k,name in enumerate(CL):
    row={**base,'split':split,'class_name':name,**{f:m[f][k] for f in ['support','predicted','precision','recall','f1']},'TP':cm[k,k],'FP':cm[:,k].sum()-cm[k,k],'FN':cm[k].sum()-cm[k,k],'dominant_confusion':js(sorted([(CL[j],int(cm[k,j])) for j in range(10) if j!=k and cm[k,j]],key=lambda x:-x[1]))};wp.writerow(row)
    for j in range(10):wc.writerow({**base,'split':split,'true_class':name,'predicted_class':CL[j],'count':cm[k,j]})
  for k,name in enumerate(CL):
   indices=[i for i in freq if labels[i]==k];we.writerow({**base,'split':'train','class_name':name,'available_unique':labels.count(k),'draws':exp[k],'unique_nuclei':unq[k],'repeated_draws':exp[k]-unq[k],'mean_exposure':exp[k]/unq[k] if unq[k] else 0,'maximum_exposure':max((freq[i] for i in indices),default=0),'unique_groups':len({train[i]['group'] for i in indices}),'provenance':'DETERMINISTIC SAMPLER REPLAY VERIFIED AGAINST ALL LOGGED CLASS/UID/GROUP COUNTS'})
  wd.writerow({**base,'split':'both','parameter_norms':js(h['parameter_norms']),'train_diagnostics':js(h['train'].get('model_diagnostics',M)),'dev_diagnostics':js(h['dev'].get('model_diagnostics',M))})
 counts={s:sum(r['split']==s for r in data) for s in ['train','dev','locked']};groups={s:len({r['group'] for r in data if r['split']==s}) for s in counts}
 ws.writerow(dict(run_id=run,experiment_group=group,hypothesis={'A':'Unique-data scaling at historical fixed10 A5','B':'Raw-image augmentation matched control','C':'Conditional cached sampling and classifier retraining'}.get(group,'See preregistered protocol'),changed={'A':'training size','B':'raw-image augmentation','C':'sampler or decoupled classifier phase; see exact config'}.get(group,'See config'),fixed='See exact config and preregistered protocol',**{f'{s}_nuclei':n for s,n in counts.items()},**{f'{s}_groups':n for s,n in groups.items()},grouping='ROI',manifest_sha256=prov['manifest_sha256'],status='COMPLETE' if result else 'RUNNING'))
 wrn.writerow(dict(run_id=run,seed=cfg.seed,config_sha256=sha(cp),config=js(cfg.to_dict()),provenance=js(prov),checkpoint_rule=cfg.optimizer.selection_mode,selected_epoch=result.get('selected_epoch','NOT YET SELECTED'),final_executed_epoch=history[-1]['epoch']))
 summary.append(dict(run_id=run,epochs=len(history),decoupled_epochs=len(decoupled),status='COMPLETE' if result else 'RUNNING',sampler_replay='PASS'))
for f in [fg,fp,fc,fe,fd,fr,fs]:f.close()
(T/'EPOCH_LEDGER_COMPLETENESS.json').write_text(json.dumps(dict(runs=summary,epoch_records=epochs_total,all_class_confusions=True,sampler_replay_verified=True),indent=2),encoding='utf-8')
print(json.dumps(dict(runs=len(summary),epochs=epochs_total,sampler_replay='PASS')))
