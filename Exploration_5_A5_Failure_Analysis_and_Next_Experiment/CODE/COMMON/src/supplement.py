import pathlib,json,hashlib,ast
import numpy as np,pandas as pd
A=pathlib.Path(r'C:\Users\Hngk\Documents\Codex\PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4');O=pathlib.Path(__file__).resolve().parents[1];M='NOT RECORDED IN ORIGINAL RUN'
C=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
used=json.loads((O/'SOURCE_HASHES.json').read_text())
def read(p):
 used[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
def save(n,r):pd.DataFrame(r).fillna(M).to_csv(O/n,index=False)
ledger=pd.read_csv(O/'NEGATIVE_EVIDENCE_LEDGER.csv').to_dict('records');ledger=[r for r in ledger if r['prompt']!=2]
for p in sorted((A/'EXPLORATION_2/ARCHITECTURES').glob('*/RESULTS/RUNS/*/metrics/summary.json')):
 su=read(p);cfg=su['config'];h=su['selected'];v=h['val'];run=p.parents[1].name
 final=read(p.parent/'final.json');hyp=p.parents[1]/'hypothesis.md'
 ledger.append(dict(prompt=2,experiment_family=cfg.get('biology',cfg.get('model','FOV/loss')),experiment_id=run,hypothesis=hyp.read_text() if hyp.exists() else run,baseline='matched P2 appearance/control (see report)',changed_variable=json.dumps(cfg),parameter_count=su.get('trainable_parameters',M),seeds=cfg.get('seed',M),folds='original300/150',epoch_budget=10,ROI_F1=v['puma']['fixed10']['macro_f1'],delta=M,uncertainty='single-run endpoint; aggregate controls in report',semantic_F1=v['macro_f1'],per_class_harm=json.dumps(v['recall']),train_val_gap=h.get('generalization_gap',M),control_result=cfg.get('biology',M),decision='historical selected endpoint, development only',reason='P2 best-epoch selection; not matched to P3 fixed epoch10',validity_limitations='reused enriched150; best epoch '+str(h['epoch']),source=str(p.relative_to(A))))
for folder,baseline in [('LORA','matched five-epoch frozen'),('BIOMASK_GUIDED','F_A3')]:
 p=A/f'EXPLORATION_4/{folder}/endpoints.csv';d=pd.read_csv(p);used[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 for family,g in d.groupby('family'):
  b=d[d.family==('frozen' if folder=='LORA' else 'F_A3')].set_index('seed');delta=g.set_index('seed').ROI_F1-b.ROI_F1
  ledger.append(dict(prompt=4,experiment_family=folder,experiment_id=family,hypothesis='restricted Q/V adaptation' if folder=='LORA' else 'clean mask classification utility',baseline=baseline,changed_variable=family,parameter_count=224474 if family=='LoRA' else 27866,seeds='17;29;43',folds='original300/150' if folder=='LORA' else 'detector-held fold0 80/76',epoch_budget=5 if folder=='LORA' else 10,ROI_F1=g.ROI_F1.mean(),delta=delta.mean(),uncertainty='see original decision JSON',semantic_F1=g.semantic_F1.mean(),per_class_harm='see report and original per_class',train_val_gap=g.gap.mean(),control_result='matched frozen / shuffled masks / Gaussian',decision='not promoted',reason='no reproducible advantage under original guards',validity_limitations='conditional enriched population, reused development',source=str(p.relative_to(A))))
al=read(A/'EXPLORATION_4/STAGE1_PROPOSALS/alignment_decision.json')
ledger.append(dict(prompt=4,experiment_family='Stage1 alignment',experiment_id='C_ALIGNMENT',hypothesis='coordinate shift limits classification',baseline='GT-centered training; Stage1 evaluation',changed_variable='Stage1-centered training',parameter_count=27866,seeds='17;29;43',folds='same detector-held fold0',epoch_budget=10,ROI_F1=M,delta=al['mean_delta'],uncertainty=json.dumps(al['ROI_bootstrap95']),semantic_F1=M,per_class_harm=json.dumps(al['class_recall_deltas']),train_val_gap=M,control_result='2x2 train/evaluation coordinates',decision='not promoted',reason='interval crosses zero; apoptosis recall -1/6',validity_limitations=al['limits']))
save('NEGATIVE_EVIDENCE_LEDGER.csv',ledger)
# Observed exposure data exist in P4 parity-admitted A0 reruns; never invent missing P3 draws.
ex=[]
for p in sorted((A/'EXPLORATION_4/RUNS').glob('A0_s*/history.jsonl')):
 used[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
 for line in p.read_text().splitlines():
  h=json.loads(line)
  ex.append(dict(run_id=p.parent.name,epoch=h['epoch'],source='P4 exact A5 parity rerun, not original P3 measurement',class_exposure=json.dumps(h['exposure']),unique_samples_seen=h.get('unique_UIDs',M),repeat_rate=h.get('repeat_rate',M),unique_ROIs_exposed=h.get('unique_ROIs',M),unique_cases_exposed=M,ROI_exposure=json.dumps(h.get('ROI_exposure',M)),effective_samples_seen=h.get('effective_samples_seen',M)))
save('A5_SAMPLING_EXPOSURE.csv',ex)
# Class-conditional permutation avoids interpreting raw high-cardinality ROI eta2 as proof.
mf=pd.read_csv(O/'SPLIT_MANIFEST_AUDIT.csv');y=mf.label.to_numpy();roi=mf.roi.to_numpy();gidx=np.unique(roi,return_inverse=True)[1];cnt=np.bincount(gidx);B=np.load(A/'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS/02_cache/biology/tierA.npy').astype(float)
def eta(v):
 means=np.bincount(gidx,weights=v)/cnt;return np.sum(cnt*(means-v.mean())**2)/max(np.sum((v-v.mean())**2),1e-12)
rng=np.random.default_rng(1701);out=[]
names=['H_mean','H_std','H_p10','H_p50','H_p90','H_gradient_mean','H_gradient_std','H_abs_laplacian_mean','H_entropy32','H_center_minus_ring','gray_center_minus_ring','H_ring_std','central_valid_fraction','ring_valid_fraction','H_robust_z','H_ROI_percentile']
for j in range(16):
 v=B[:,j].copy()
 for c in range(10):v[y==c]-=v[y==c].mean()
 actual=eta(v);null=[]
 for k in range(499):
  z=v.copy()
  for c in range(10):z[y==c]=rng.permutation(v[y==c])
  null.append(eta(z))
 out.append(dict(feature=names[j],class_residual_ROI_eta2=actual,class_conditional_null_mean=np.mean(null),excess_eta2=actual-np.mean(null),exploratory_permutation_p=(1+sum(np.array(null)>=actual))/500,permutations=499,interpretation='class-conditioned association; not biological causality; correlated features, multiple testing'))
save('TIER_A_ROI_PERMUTATION.csv',out)
# Correctness correlation uses one OOF prediction per seed/cell, then averages seeds per cell.
correct=np.zeros((3,450));correct[:]=np.nan
for si,seed in enumerate([17,29,43]):
 for p in (A/'EXPLORATION_3/ARCHITECTURES/A5/RESULTS/RUNS').glob(f'*real_s{seed}_cv*/predictions.npz'):
  z=np.load(p);correct[si,z['indices']]=(z['logits'].argmax(1)==z['labels'])
ix=mf.split.eq('train').to_numpy();avg=np.mean(correct[:,ix],0)
co=[]
for j in range(16):
 v=B[ix,j];res=v.copy();cr=avg.copy()
 for c in range(10):res[y[ix]==c]-=v[y[ix]==c].mean();cr[y[ix]==c]-=avg[y[ix]==c].mean()
 co.append(dict(feature=names[j],OOF_correctness_correlation=np.corrcoef(v,avg)[0,1],class_residual_correctness_correlation=np.corrcoef(res,cr)[0,1]))
save('TIER_A_CORRECTNESS_AUDIT.csv',co)
# Proposal coverage and the upstream-exclusion obstacle.
p=A/'EXPLORATION_4/STAGE1_PROPOSALS/proposal_manifest.csv';pr=pd.read_csv(p);used[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
save('STAGE1_PROPOSAL_COUNTS.csv',pr.groupby(['roi_id','fold','match_status']).size().rename('count').reset_index())
fold_values=sorted(pr.fold.unique().tolist());training_sets=[set(json.loads(x)) for x in pr.source_training_folds.unique()]
clean_folds=[f for f in fold_values if all(f not in s for s in training_sets)]
admission=dict(total_proposals=len(pr),matched=int(sum(pr.match_status=='matched')),unmatched=int(sum(pr.match_status!='matched')),ROIs=pr.roi_id.nunique(),zero_proposal_ROIs=0,folds=fold_values,fold_excluded_by_every_fixed_source_detector=clean_folds,complete_population_outer_independence_admitted=bool(clean_folds),reason='Every historical fold occurs in training of other source detectors. Row-wise OOF is insufficient for the requested exclusion of all Stage2 outer-held groups from upstream dependencies. No existing complete-population four-way clean split is certified.',patient_or_case_mapping='unverified; ROI proxy only',locked_test='absent; do not relabel reused development',full_data_A5_executed=False)
(O/'FULL_BASELINE_ADMISSION.json').write_text(json.dumps(admission,indent=2))
(O/'SOURCE_HASHES.json').write_text(json.dumps(used,indent=2))
print('Supplement complete; ledger',len(ledger),'exposure rows',len(ex),'proposal admission',admission)
