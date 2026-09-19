"""Read-only historical diagnostics; Python + numpy + pandas, no training."""
import json, pathlib, hashlib, re, time, sys, html
import numpy as np
import pandas as pd
A=pathlib.Path(r'C:\Users\Hngk\Documents\Codex\PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4')
O=pathlib.Path(__file__).resolve().parents[1]
M='NOT RECORDED IN ORIGINAL RUN'
C=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
used={}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def track(p):used[str(p)]=sha(p);return p
def js(p):return json.loads(track(p).read_text(encoding='utf-8-sig'))
def csv(p):return pd.read_csv(track(p))
def save(name,rows):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
 d.fillna(M).to_csv(O/name,index=False);return d
def dump(name,x): (O/name).write_text(json.dumps(x,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding='utf-8')
def roi(v):return v.get('puma',{}).get('fixed10',{}).get('macro_f1',np.nan)
def hist(p):return [json.loads(x) for x in track(p).read_text().splitlines() if x.strip()]
start=time.time()
files=[p for p in A.rglob('*') if p.is_file() and '.git' not in p.parts]
save('ARCHIVE_INVENTORY.csv',[{'path':str(p.relative_to(A)),'bytes':p.stat().st_size} for p in files])
print('Inventory',len(files),flush=True)
mf=csv(A/'EXPLORATION_3/ARCHITECTURES/A5/INPUT_MANIFESTS/sample_manifest.csv')
cv=csv(A/'EXPLORATION_3/ARCHITECTURES/A5/INPUT_MANIFESTS/prompt3_cv.csv')
assert mf.uid.tolist()==cv.uid.tolist()
mf['cv_fold']=cv.cv_fold
assert mf.groupby('roi').split.nunique().max()==1
assert mf[mf.split=='train'].groupby('roi').cv_fold.nunique().max()==1
save('SPLIT_MANIFEST_AUDIT.csv',mf)
dyn=[];pc=[];peaks=[];end=[]
for p in sorted((A/'EXPLORATION_3/ARCHITECTURES/A5/RESULTS/RUNS').glob('*/history.jsonl')):
 if not ('A5_real_' in str(p) or 'confirm_A5' in str(p)):continue
 cfg=js(p.parent/'config.json'); hh=hist(p);run=p.parent.name
 seed=cfg['seed'];fold=cfg.get('fold','confirmation');scope='confirmation' if 'confirm' in run else 'CV'
 for h in hh:
  t,v=h['train'],h['val'];di=h.get('diagnostics',{})
  row=dict(run_id=run,seed=seed,fold=fold,scope=scope,epoch=h['epoch'],train_loss=h.get('train_CE',M),val_loss=h.get('val_CE',M),sampled_objective=h.get('objective',M),train_macro_f1=t['macro_f1'],val_macro_f1=v['macro_f1'],ROI_F1=roi(v),balanced_accuracy=v['balanced_accuracy'],accuracy=v['accuracy'],NLL=v['nll'],ECE15=v['ece15'],train_val_gap=h.get('gap',t['macro_f1']-v['macro_f1']),learning_rate=h.get('lr',M),class_exposure=json.dumps(h.get('exposure',M)),gradient_norms=json.dumps(h.get('positive_target_head_gradient',M)),gradient_scope=h.get('gradient_scope',M),source=str(p.relative_to(A)))
  for k in ['interaction_norm','appearance_projection_norm','biology_projection_norm','correction_norm']:
   row[k]=float(np.mean(di[k])) if k in di else M
   row[k+'_raw']=json.dumps(di.get(k,M))
  dyn.append(row)
  for c,n in enumerate(C):
   cm=np.array(v['confusion'])[c].copy();cm[c]=0
   pc.append(dict(run_id=run,seed=seed,fold=fold,scope=scope,epoch=h['epoch'],class_name=n,precision=v['precision'][c],recall=v['recall'][c],F1=v['f1'][c],support=v['support'][c],predicted_count=v['predicted'][c],confusion_destinations=json.dumps({C[j]:int(x) for j,x in enumerate(cm) if x}),dominant_wrong_class=C[int(cm.argmax())] if cm.max()>0 else 'none'))
 end.append(dyn[-1])
 for k,fun in [('semantic_F1',lambda h:h['val']['macro_f1']),('ROI_F1',lambda h:roi(h['val']))]:
  values=[fun(h) for h in hh];best=int(np.argmax(values))
  peaks.append(dict(run_id=run,scope=scope,metric=k,peak_epoch=hh[best]['epoch'],peak=values[best],final=values[-1],peak_to_final_degradation=values[best]-values[-1],selection_status='retrospective diagnostic only'))
d=save('A5_LEARNING_DYNAMICS.csv',dyn);per=pd.DataFrame(pc)
for (run,c),g in per.groupby(['run_id','class_name']):
 ix=(per.run_id==run)&(per.class_name==c)
 for metric in ['recall','F1']:
  best=g.loc[g[metric].idxmax()]
  per.loc[ix,'epoch_of_best_'+metric]=best.epoch
  per.loc[ix,'epoch10_change_from_best_'+metric]=g.iloc[-1][metric]-best[metric]
final=per[per.epoch==10]
for c in C:
 g=final[(final.class_name==c)&(final.scope=='CV')]
 for metric in ['recall','F1']:
  per.loc[per.class_name==c,metric+'_fold_mean_sd']=g.groupby('fold')[metric].mean().std()
  per.loc[per.class_name==c,metric+'_seed_mean_sd']=g.groupby('seed')[metric].mean().std()
save('A5_PER_CLASS_STABILITY.csv',per);save('A5_PEAK_VS_FINAL.csv',peaks)
save('A5_ENDPOINTS.csv',end)
assert len(d)==120 and len(d[d.scope=='CV'])==90
print('A5 epochs',len(d),'CV final means',d[(d.scope=='CV')&(d.epoch==10)][['train_macro_f1','val_macro_f1','train_val_gap','NLL','ECE15']].mean().to_dict(),flush=True)
# Complete historical run-endpoint ledger, using canonical architecture copies and no duplicated histories.
ledger=[]
for prompt in [2,3,4]:
 roots=list((A/f'PROMPT_{prompt}/ARCHITECTURES').glob('*/RESULTS/RUNS/*/history.jsonl')) if prompt<4 else list((A/'EXPLORATION_4/RUNS').glob('*/history.jsonl'))
 for p in roots:
  hh=hist(p);h=hh[-1];cp=p.parent/'config.json';cfg=js(cp) if cp.exists() else {}
  v=h.get('val',{});t=h.get('train',{});sp=p.parent/'summary.json';su=js(sp) if sp.exists() else {}
  family=cfg.get('architecture',cfg.get('family',p.parent.name.split('_s')[0]));run=p.parent.name
  ledger.append(dict(prompt=prompt,experiment_family=family,experiment_id=run,hypothesis=cfg.get('hypothesis',family),baseline='Exploration2 A1' if prompt==3 else ('Exploration3 A5' if prompt==4 else 'matched historical control'),changed_variable=json.dumps(cfg),parameter_count=su.get('parameters',su.get('trainable_parameters',M)),seeds=cfg.get('seed',M),folds=cfg.get('fold',M),epoch_budget=len(hh),ROI_F1=roi(v),delta=M,uncertainty=M,semantic_F1=v.get('macro_f1',M),per_class_harm=json.dumps(v.get('recall',M)),train_val_gap=h.get('gap',M),control_result=cfg.get('control',M),decision='historical; see aggregate family ledger',reason='fixed endpoint; no new rerun',validity_limitations='enriched development; ROI grouping only; full-proposal evaluation absent',source=str(p.relative_to(A))))
save('HISTORICAL_RUN_ENDPOINTS.csv',ledger)
p3=csv(A/'EXPLORATION_3/UNRESOLVED_HISTORICAL_ARTIFACTS/outputs/STAGE2_RESEARCH_20260907/90_comparative_analysis/architecture_seed_summary.csv')
agg=[]
for _,r in p3.iterrows():
 agg.append(dict(prompt=3,experiment_family=r.architecture,experiment_id=r.architecture+'_'+r.control,hypothesis='test '+r.architecture+' with '+r.control,baseline='A1',changed_variable=r.control,parameter_count=r.parameters,seeds='17;29;43',folds='0;1;2',epoch_budget=10,ROI_F1=r.ROI_F1_mean,delta=r.paired_delta_vs_A1,uncertainty=json.dumps([r.roi_bootstrap_low,r.roi_bootstrap_high]),semantic_F1=r.semantic_f1_mean,per_class_harm='see HISTORICAL_RUN_ENDPOINTS.csv',train_val_gap=r.gap_mean,control_result=r.control,decision=r.decision,reason='predeclared gain/control/class/gap/uncertainty rule',validity_limitations='upstream contaminated; diagnostic only' if r.architecture.startswith('B') else '300 enriched nuclei; reused confirmation; ROI grouping'))
for table in ['A_complete','D_complete']:
 for _,r in csv(A/f'EXPLORATION_4/TABLES/{table}/seed_summary.csv').iterrows():
  agg.append(dict(prompt=4,experiment_family=table,experiment_id=r.family,hypothesis='representation/multiscale/sampling controlled comparison',baseline='P3 A5 / P4 A0',changed_variable=r.family,parameter_count=r.parameters,seeds='17;29;43',folds='0;1;2',epoch_budget=10,ROI_F1=r.ROI_F1_mean,delta=r.delta_vs_A5,uncertainty=json.dumps([r.bootstrap_low,r.bootstrap_high]),semantic_F1=M,per_class_harm='see raw per_class.csv and run histories',train_val_gap=M,control_result=r.control_advantage,decision='internal eligible; A3 final confirmation FAILED' if r.promoted else 'not promoted',reason='finite predeclared promotion rule',validity_limitations='300 enriched development nuclei; no untouched test'))
agg.extend(x for x in ledger if x['prompt']==2)
save('NEGATIVE_EVIDENCE_LEDGER.csv',agg)
# Cache admission: representation output hashes, identical CLS to historical feature cache, matching manifests.
repdir=A/'EXPLORATION_4/TARGET_POOLING/REPRESENTATIONS';contract=js(repdir/'complete.json');checks=[];X={}
for n,key in [('target_patch','A1'),('gaussian','A2'),('neighborhood','A3')]:
 p=repdir/(key+'.npy');actual=sha(p);expected=contract['files'][key]
 assert actual==expected,(key,'hash mismatch')
 X[n]=np.load(track(p),allow_pickle=False);checks.append(dict(cache=str(p),expected_sha256=expected,actual_sha256=actual,pass_check=True))
p=A/'EXPLORATION_4/TOKEN_AUDIT/CACHE/cls.npy';X['CLS']=np.load(track(p),allow_pickle=False)
old=A/'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS/02_cache/uni2_features/gt_fov96/features.npy'
assert np.array_equal(X['CLS'],np.load(track(old),allow_pickle=False))
checks.append(dict(cache=str(p),expected_sha256=sha(old),actual_sha256=sha(p),pass_check=True,comparison='array equality to P2 CLS; file headers may differ'))
save('CACHE_IDENTITY_AUDIT.csv',checks)
y=mf.label.to_numpy();g=mf.roi.to_numpy();folds=mf.cv_fold.to_numpy();tr=mf.split.eq('train').to_numpy();va=~tr
geom=[];pairs=[]
def norm(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def covdist(a,b):
 a=a-a.mean(0);b=b-b.mean(0)
 aa=np.sum((a@a.T)**2)/max(len(a)-1,1)**2;bb=np.sum((b@b.T)**2)/max(len(b)-1,1)**2
 ab=np.sum((a@b.T)**2)/max(len(a)-1,1)/max(len(b)-1,1)
 return float(np.sqrt(max(aa+bb-2*ab,0)))
for name,raw in X.items():
 assert raw.shape==(450,1536) and np.isfinite(raw).all()
 x=norm(raw.astype(float));sim=x@x.T
 same=y[:,None]==y;rg=g[:,None]==g;off=~np.eye(len(y),dtype=bool)
 centers=np.stack([x[tr&(y==c)].mean(0) for c in range(10)]);center_unit=norm(centers)
 overall=x[tr].mean(0);between=sum(sum(tr&(y==c))*np.sum((centers[c]-overall)**2) for c in range(10));within=sum(np.sum((x[tr&(y==c)]-centers[c])**2) for c in range(10))
 for c,n in enumerate(C):
  a=tr&(y==c);b=va&(y==c);s=(y==c)[:,None]&same&off
  margins=[];knn=[];sil=[]
  for f in [0,1,2]:
   train=tr&(folds!=f);test=tr&(folds==f)&(y==c);ii=np.flatnonzero(test);jj=np.flatnonzero(train)
   cc=norm(np.stack([x[train&(y==k)].mean(0) for k in range(10)]));sc=x[test]@cc.T
   other=sc.copy();other[:,c]=-np.inf;margins.extend(sc[:,c]-other.max(1))
   if len(ii):
    near=jj[np.argsort(-sim[np.ix_(ii,jj)],axis=1)[:,:5]];knn.extend(np.mean(y[near]==c,axis=1))
    own=1-sim[np.ix_(ii,np.flatnonzero(train&(y==c)))].mean(1)
    oth=np.min(np.stack([1-sim[np.ix_(ii,np.flatnonzero(train&(y==k)))].mean(1) for k in range(10) if k!=c]),axis=0)
    sil.extend((oth-own)/np.maximum(np.maximum(own,oth),1e-12))
  simtr=x[a]@center_unit.T;simva=x[b]@center_unit.T
  def margin(z):
   z=z.copy();v=z[:,c].copy();z[:,c]=-np.inf;return float(np.mean(v-z.max(1)))
  def entropy(z):
   # Fixed temperature 1 cosine pseudo-distribution, not calibrated model probability.
   p=np.exp(z-z.max(1,keepdims=True));p/=p.sum(1,keepdims=True);return float(np.mean(-np.sum(p*np.log(p),1)))
  competing=centers@centers[c]/np.maximum(np.linalg.norm(centers,axis=1)*np.linalg.norm(centers[c]),1e-12);competing[c]=-np.inf
  geom.append(dict(representation=name,class_name=n,n_train=int(a.sum()),n_confirmation=int(b.sum()),within_class_cosine=float(sim[s].mean()),between_class_cosine=float(sim[(y==c)[:,None]&~same].mean()),same_class_same_ROI_cosine=float(sim[s&rg].mean()) if (s&rg).any() else np.nan,same_class_different_ROI_cosine=float(sim[s&~rg].mean()),same_class_different_case_cosine=M,Fisher_between_within=between/within,held_fold_5NN_purity=float(np.mean(knn)),held_fold_centroid_margin=float(np.mean(margins)),held_fold_silhouette_like=float(np.mean(sil)),train_confirmation_centroid_shift=float(np.linalg.norm(x[a].mean(0)-x[b].mean(0))),covariance_Frobenius_shift=covdist(x[a],x[b]),nearest_competing_class=C[int(competing.argmax())],train_margin=margin(simtr),confirmation_margin=margin(simva),margin_change=margin(simva)-margin(simtr),cosine_softmax_entropy_change=entropy(simva)-entropy(simtr),scope='descriptive; confirmation reused; no selection'))
  for k in range(10):pairs.append(dict(representation=name,class_name=n,other_class=C[k],centroid_cosine=float(center_unit[c]@center_unit[k]),centroid_distance=float(np.linalg.norm(centers[c]-centers[k]))))
save('REPRESENTATION_GEOMETRY.csv',geom);save('REPRESENTATION_CENTROIDS.csv',pairs)
print('Geometry complete',flush=True)
# Exact confirmation confusion changes, validate UID/label alignment before subtraction.
conf=[]
for seed in [17,29,43]:
 bp=A/f'EXPLORATION_3/ARCHITECTURES/A5/RESULTS/RUNS/p3_confirm_A5_s{seed}/predictions.npz'
 candidates=list((A/'EXPLORATION_4/RUNS').glob(f'FINAL_A3_s{seed}/epoch_10_predictions.npz'))
 if not candidates:candidates=list((A/'EXPLORATION_4/ARCHITECTURES/FINAL_A3/RESULTS/RUNS').glob(f'*/predictions.npz'))
 lp=[q for q in candidates if f's{seed}' in str(q)][0]
 bz=np.load(track(bp),allow_pickle=False);lz=np.load(track(lp),allow_pickle=False)
 assert np.array_equal(bz['indices'],lz['indices']) and np.array_equal(bz['labels'],lz['labels'])
 yy=bz['labels'];bh=bz['logits'].argmax(1);lh=lz['logits'].argmax(1)
 for c,n in enumerate(C):
  for k,dest in enumerate(C):
   conf.append(dict(seed=seed,true_class=n,predicted_class=dest,support=int(sum(yy==c)),CLS_count=int(sum((yy==c)&(bh==k))),local_count=int(sum((yy==c)&(lh==k))),delta_count=int(sum((yy==c)&(lh==k))-sum((yy==c)&(bh==k)))))
save('CONFUSION_CHANGES.csv',conf)
# Nuclei census from raw GeoJSON features; do not confuse polygon components with features.
data=pathlib.Path(r'D:\Research\PUMA\Code\TRAINING CODE\Dataset\01_training_dataset_geojson_nuclei');full=[]
for p in sorted(data.glob('*.geojson')):
 z=js(p)
 for i,f in enumerate(z['features']):
  props=f.get('properties',{});label=props.get('classification',{});label=label.get('name','') if isinstance(label,dict) else str(label)
  label=label.removeprefix('nuclei_')
  if label in C:full.append(dict(uid=p.stem.removesuffix('_nuclei')+':'+str(i),roi=p.stem.removesuffix('_nuclei'),class_name=label))
full=pd.DataFrame(full);div=[]
for population,df in [('full_annotation_census',full),('development450',mf),('train300',mf[tr]),('reused_confirmation150',mf[va])]:
 for c in C:
  z=df[df.class_name==c];counts=z.groupby('roi').size();n=len(z)
  div.append(dict(population=population,class_name=c,nuclei=n,ROIs=len(counts),patients_cases=M,slides=M,median_nuclei_per_ROI=float(counts.median()),max_percentage_one_ROI=100*counts.max()/n,effective_positive_ROIs=n*n/float(sum(counts**2)),train_support=int(sum(tr&(y==C.index(c)))) if population=='development450' else M,validation_support=int(sum(va&(y==C.index(c)))) if population=='development450' else M,test_support=M))
save('DATA_DIVERSITY_AUDIT.csv',div);save('FULL_CENSUS_ROI_CLASS_COUNTS.csv',full.groupby(['roi','class_name']).size().rename('nuclei').reset_index())
print('Census',len(full),'ROIs',full.roi.nunique(),flush=True)
# Tier-A descriptive eta-squared with label-adjusted ROI residual association; no causal interpretation.
bpath=A/'EXPLORATION_2/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS/02_cache/biology/tierA.npy';bio=np.load(track(bpath),allow_pickle=False);assert bio.shape==(450,16)
def eta(v,labels):
 total=sum((v-v.mean())**2)
 return sum(sum(labels==l)*(v[labels==l].mean()-v.mean())**2 for l in np.unique(labels))/max(total,1e-12)
aud=[]
for j in range(16):
 v=bio[:,j].astype(float);res=v.copy()
 for c in range(10):res[y==c]-=v[y==c].mean()
 border=np.minimum.reduce([mf.x.to_numpy(),mf.y.to_numpy(),1024-mf.x.to_numpy(),1024-mf.y.to_numpy()])
 aud.append(dict(feature_index=j,class_eta2=eta(v,y),ROI_eta2=eta(v,g),class_residual_ROI_eta2=eta(res,g),split_eta2=eta(v,tr),primary_metastatic_eta2=eta(v,np.array(['metastatic' in r for r in g])),border_distance_Pearson=float(np.corrcoef(v,border)[0,1]),patient_case_association=M,acquisition_metadata_association=M,scope='descriptive; ROI eta2 biased by many groups; border assumes verified 1024 ROI contract'))
save('TIER_A_CONFOUNDING_AUDIT.csv',aud)
dump('ANALYSIS_SUMMARY.json',dict(A5_CV_epoch1=d[(d.scope=='CV')&(d.epoch==1)].select_dtypes('number').mean().to_dict(),A5_CV_epoch10=d[(d.scope=='CV')&(d.epoch==10)].select_dtypes('number').mean().to_dict(),geometry=pd.DataFrame(geom).groupby('representation').mean(numeric_only=True).to_dict('index'),full_nuclei=len(full),full_ROIs=full.roi.nunique(),runtime_seconds=time.time()-start))
dump('SOURCE_HASHES.json',used)
print('Analysis finished',time.time()-start,flush=True)
