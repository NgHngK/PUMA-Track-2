import json,csv,hashlib,shutil,re,sys
from pathlib import Path
import numpy as np,torch
from sklearn.model_selection import StratifiedGroupKFold
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';A=R/'00_reference/biomask_audit';C=R/'01_shared_core'
S=Path('D:/Research/PUMA/Code/Version 17/PUMA_Nuclei_Pipeline/src/puma_nuclei')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2),encoding='utf-8')
def csvout(p,rs):
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
src=A/'source';src.mkdir(exist_ok=True)
names=['biomask.py','biomask_data.py','biomask_sampling.py','biomask_trainer.py','biology_features.py','crops.py','targets.py','preprocess.py']
for n in names:shutil.copy2(S/'stage2'/n,src/n)
shutil.copy2(S/'config.py',src/'config.py');shutil.copy2(S/'constants.py',src/'constants.py')
for n in ['folds.npy','roi_manifest.npy','biomask_provenance.json','biomask_cache_contract.json','drive_inventory.json']:shutil.copy2(W/n,A/n)
cks=A/'checkpoints';cks.mkdir(exist_ok=True);idx=[]
prov=json.loads((W/'biomask_provenance.json').read_text())['fold_sources']
for f in range(5):
    p=W/f'fold{f}.pt';s=torch.load(p,weights_only=True,map_location='cpu');e=s['extra'];h=sha(p)
    assert e['fold']==f and f not in e['training_folds'];assert h==next(v for v in prov if v['fold']==f)['checkpoint_sha256']
    shutil.copy2(p,cks/p.name);dump(A/f'checkpoint_metadata_fold{f}.json',e)
    idx.append({'fold':f,'path':str((cks/p.name).relative_to(R)),'sha256':h,'epoch_zero_based':e['epoch'],'selection':'final epoch; not minimum held loss','training_folds':str(e['training_folds']),'parameters':sum(v.numel() for v in s['model'].values()),'provenance_hash_match':True})
csvout(A/'checkpoint_index.csv',idx)
roi=np.load(W/'roi_manifest.npy');fold=np.load(W/'folds.npy');mapping=dict(zip(roi['roi_id'],map(int,fold)))
rows=list(csv.DictReader((R/'01_sample_definition/sample_manifest.csv').open()));train=np.array([r['split']=='train' for r in rows]);y=np.array([int(r['label']) for r in rows]);groups=np.array([r['group'] for r in rows]);indices=np.flatnonzero(train)
cv=np.full(len(rows),-1);sg=StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=17)
for f,(ti,vi) in enumerate(sg.split(indices,y[indices],groups[indices])):
    cv[indices[vi]]=f;assert set(groups[indices[ti]]).isdisjoint(groups[indices[vi]]);assert len(np.unique(y[indices[ti]]))==10
split=[{'uid':r['uid'],'roi':r['roi'],'original_split':r['split'],'cv_fold':int(cv[i]),'historical_biomask_fold':mapping[r['roi']]} for i,r in enumerate(rows)]
csvout(R/'01_sample_definition/prompt3_cv.csv',split)
leaks=[]
for outer in ['original_val',0,1,2]:
    held=[r for i,r in enumerate(rows) if (r['split']=='val' if outer=='original_val' else cv[i]==outer)]
    hf=set(mapping[r['roi']] for r in held)
    for f in range(5):
        trained=set(range(5))-{f};leaks.append({'outer':outer,'checkpoint_fold':f,'BioMask_training_folds':str(sorted(trained)),'held_historical_folds':str(sorted(hf)),'outer_annotation_overlap':str(sorted(hf&trained)),'outer_isolation':not bool(hf&trained),'row_OOF_is_not_nested':True})
csvout(A/'fold_provenance.csv',leaks)
history=[]
for f,e,tr,va in re.findall(r'BioMask fold=(\d+) epoch=(\d+)/30 train=([\d.]+) held=([\d.]+)',(A/'training_log.txt').read_text()):history.append({'fold':int(f),'epoch':int(e),'train':float(tr),'held':float(va)})
csvout(A/'training_history.csv',history)
dump(A/'historical_fold_support.json',{'roi_counts':np.bincount(fold,minlength=5).tolist(),'case_id_equals_roi_id':bool(np.all(roi['case_id']==roi['roi_id'])),'note':'case_id is an ROI identifier, not independently verified patient metadata'})
dump(A/'biomask_schema.json',{'input':'RGB [0,1] + Gaussian prompt sigma2.5; 96 source pixels; reflect padding and valid mask','internal_channels':['R','G','B','H proxy','forward H gradient','prompt'],'base_channels':32,'outputs':['mask_logits','presence_logits','quality_logits','center_offset','presence_local_support'],'offset_range':[-24,24],'GT_class_input':False,'training_loss':'.52*(instance(.6 BCE+.4 softDice)+.25 emptyBCE)+.23 presenceBCE+.17 smoothL1offset+.08 qualityBCE','mask_quality_target':'current hard-mask IoU against GT; diagnostic supervision only','selected_checkpoint':'final epoch30','downstream_status':'exploratory contaminated upstream features only; never promotion eligible'})
csvout(A/'artifact_index.csv',[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in A.rglob('*') if p.is_file()])
state=json.loads((R/'00_reference/RESEARCH_STATE.json').read_text());state.update(status='active',phase='Exploration3 BioMask recovered, audit and predeclared architecture CV',exploration3={'cv':'300 prior training features, 3 ROI-grouped folds; original150 held aside','biomask':'five finals recovered and hashes verified; outer contaminated','next':'extract diagnostic TierB then finite CV controls'});dump(R/'00_reference/RESEARCH_STATE.json',state)
print('CV support',[(f,np.bincount(y[cv==f],minlength=10).tolist()) for f in range(3)]);print('historical folds',np.bincount(fold));print('clean paths',sum(x['outer_isolation'] for x in leaks))
