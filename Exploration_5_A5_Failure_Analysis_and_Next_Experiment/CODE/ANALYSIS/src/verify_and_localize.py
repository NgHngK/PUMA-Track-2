import pathlib,json,hashlib
import numpy as np,pandas as pd
O=pathlib.Path(__file__).resolve().parents[1];A=O.parent/'PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4';mf=pd.read_csv(O/'SPLIT_MANIFEST_AUDIT.csv')
contract=json.loads((A/'EXPLORATION_4/TOKEN_AUDIT/CACHE/contract.json').read_text());assert contract['UIDs']==mf.uid.tolist();assert contract['FOV']==96
p4=pd.read_csv(A/'EXPLORATION_4/ARCHITECTURES/A3/INPUT_MANIFESTS/sample_manifest.csv');assert p4.uid.tolist()==mf.uid.tolist();assert np.array_equal(p4.label,mf.label)
rows=[]
for seed in [17,29,43]:
 b=np.load(A/f'EXPLORATION_3/ARCHITECTURES/A5/RESULTS/RUNS/p3_confirm_A5_s{seed}/predictions.npz');l=np.load(A/f'EXPLORATION_4/RUNS/FINAL_A3_s{seed}/epoch_10_predictions.npz');ix=b['indices'];assert np.array_equal(ix,l['indices'])
 for i,bc,lc in zip(ix,b['logits'].argmax(1),l['logits'].argmax(1)):
  row=mf.iloc[i];rows.append(dict(seed=seed,uid=row.uid,roi=row.roi,class_name=row.class_name,CLS_correct=int(bc==row.label),local_correct=int(lc==row.label),delta=int(lc==row.label)-int(bc==row.label)))
d=pd.DataFrame(rows);d.to_csv(O/'CONFIRMATION_CELL_EVENTS.csv',index=False)
g=d.groupby(['class_name','roi']).agg(n_events=('uid','size'),unique_nuclei=('uid','nunique'),CLS_correct=('CLS_correct','sum'),local_correct=('local_correct','sum'),delta_correct=('delta','sum')).reset_index();g.to_csv(O/'CONFIRMATION_ROI_HARM.csv',index=False)
pr=pd.read_csv(A/'EXPLORATION_4/STAGE1_PROPOSALS/proposal_manifest.csv');n=np.load(A/'EXPLORATION_4/STAGE1_PROPOSALS/nuclei.npy');ri=np.load(A/'EXPLORATION_4/STAGE1_PROPOSALS/roi_manifest.npy');rm={str(x['roi_id']):i for i,x in enumerate(ri)}
match=pr[pr.match_status=='matched'];print('matched_gt_uid range',match.matched_gt_uid.min(),match.matched_gt_uid.max(),'duplicates within roi',match.duplicated(['roi_id','matched_gt_uid']).sum())
# GT id in proposal manifest is global nuclei-array index (verify against roi_index).
gtids=match.matched_gt_uid.astype(int).to_numpy();assert np.all(n['roi_index'][gtids]==match.roi_id.map(rm).to_numpy())
miss=np.ones(len(n),bool);miss[np.unique(gtids)]=False
pd.DataFrame([{'roi':str(ri[x['roi_index']]['roi_id']),'GT_component_array_index':int(i),'V17_class_id':int(x['class_id']),'x':float(x['x']),'y':float(x['y'])} for i,x in enumerate(n) if miss[i]]).to_csv(O/'MISSED_GT_COMPONENTS.csv',index=False)
used=json.loads((O/'SOURCE_HASHES.json').read_text());changed=[]
for path,expected in used.items():
 actual=hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
 if actual!=expected:changed.append(path)
assert not changed
z=dict(cache_UID_contract_pass=True,canonical_labels_pass=True,consumed_source_hashes_unchanged=len(used),changed_sources=changed,matched_unique_GT_components=len(np.unique(gtids)),missed_GT_components=int(miss.sum()),feature_census_not_substituted_for_component_population=True)
(O/'VERIFICATION.json').write_text(json.dumps(z,indent=2));print(z)
