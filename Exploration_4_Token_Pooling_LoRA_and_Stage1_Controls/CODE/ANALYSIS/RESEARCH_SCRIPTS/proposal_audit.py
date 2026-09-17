import sys,json,csv,hashlib,collections,importlib.util
sys.dont_write_bytecode=True
from bootstrap import *
import numpy as np
from scipy.spatial import cKDTree
S=Path('D:/Research/PUMA/Code/Version 17/PUMA_Nuclei_Pipeline/src/puma_nuclei');D=R/'EXPLORATION_4/STAGE1_PROPOSALS'
if __name__=='__main__':
 p=np.load(W/'proposals.npy');g=np.load(W/'nuclei.npy');rm=np.load(BASE/'work/exploration3/roi_manifest.npy');folds=np.load(BASE/'work/exploration3/folds.npy');summary=json.loads((W/'proposal_summary_decoded.json').read_text());bc=json.loads((BASE/'work/exploration3/biomask_cache_contract.json').read_text())
 spec=importlib.util.spec_from_file_location('source_provenance',S/'utils/provenance.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 assert mod.numpy_array_sha256(p)==summary['proposal_sha256'];assert hashlib.sha256(p.tobytes()).hexdigest()==bc['proposal_sha256']
 for src,name in [(W/'proposals.npy','proposals.npy'),(W/'nuclei.npy','nuclei.npy'),(W/'proposal_summary_decoded.json','source_summary.json'),(W/'stage1_config_decoded.json','stage1_config.json'),(BASE/'work/exploration3/roi_manifest.npy','roi_manifest.npy'),(BASE/'work/exploration3/folds.npy','folds.npy'),(S/'stage1/inference.py','stage1_inference_source.py.txt'),(S/'utils/provenance.py','hash_implementation.py.txt')]:cp(src,D/name)
 gi={int(v['nucleus_index']):v for v in g};ops={v['fold']:v for v in summary['fold_operating_points']};records=[];distances=[]
 for v in p:
  matched=int(v['matched_gt_index'])>=0;gt=gi[int(v['matched_gt_index'])] if matched else None;f=int(v['fold']);roi=int(v['roi_index']);assert int(folds[roi])==f
  distance=float(np.hypot(v['x']-gt['x'],v['y']-gt['y'])) if matched else None
  if matched:assert gt['roi_index']==roi and gt['class_id']==v['class_id'];distances.append(distance)
  records.append({'proposal_uid':str(v['proposal_uid']),'roi_id':str(rm[roi]['roi_id']),'patient_id':'UNKNOWN; case_id equals ROI','fold':f,'x':float(v['x']),'y':float(v['y']),'detector_score':float(v['heatmap_score']*v['quality']),'heatmap_score':float(v['heatmap_score']),'quality':float(v['quality']),'match_status':'matched' if matched else 'unmatched','matched_gt_uid':int(v['matched_gt_index']),'semantic_target_V17':int(v['class_id']),'distance_to_gt':distance,'source_checkpoint':ops[f]['checkpoint'],'source_checkpoint_sha256_claimed':ops[f]['checkpoint_sha256'],'source_training_folds':str(ops[f]['training_folds']),'proposal_threshold':ops[f]['threshold'],'NMS':ops[f]['suppression_radius_pixels'],'hash_contract_verified':True,'outer_independence':'mixed current outer folds: NOT independent'})
 csvout(D/'proposal_manifest.csv',records)
 sys.path.insert(0,str(OLD/'01_shared_core'));from dataset import read_manifest
 rows=read_manifest(OLD/'01_sample_definition/sample_manifest.csv');roiidx={str(v['roi_id']):i for i,v in enumerate(rm)};pairs=[]
 canonical_to_v17=[0,1,2,3,4,5,6,8,7,9];matchedmap={int(v['matched_gt_index']):v for v in p if v['matched_gt_index']>=0}
 for i,r in enumerate(rows):
  roi=roiidx[r['roi']];gg=g[g['roi_index']==roi];xy=np.array([[v['x'],v['y']] for v in gg]);ds,ind=cKDTree(xy).query([float(r['eval_x']),float(r['eval_y'])]);gt=gg[ind]
  if ds>1e-3 or int(gt['class_id'])!=canonical_to_v17[r['label']]:
   pairs.append({'index':i,'uid':r['uid'],'status':'unresolved_GT_component','source_GT_distance':float(ds)});continue
  v=matchedmap.get(int(gt['nucleus_index']))
  pairs.append({'index':i,'uid':r['uid'],'status':'matched' if v is not None else 'missed','historical_fold':int(folds[roi]),'gt_source_index':int(gt['nucleus_index']),'source_GT_distance':float(ds),'stage1_x':float(v['x']) if v is not None else None,'stage1_y':float(v['y']) if v is not None else None,'proposal_uid':str(v['proposal_uid']) if v is not None else None,'class':r['label'],'displacement_from_area_centroid':float(np.hypot(v['x']-r['x'],v['y']-r['y'])) if v is not None else None})
 dump(D/'paired450.json',pairs)
 dd=np.array(distances);q={'proposal_count':len(p),'GT_components':len(g),'GT_features_census':97193,'matched':len(dd),'unmatched':int((p['is_reject']==1).sum()),'zero_proposal_ROIs':len(set(range(205))-set(p['roi_index'])),'mean_displacement':float(dd.mean()),'median':float(np.median(dd)),'p90':float(np.quantile(dd,.9)),'p95':float(np.quantile(dd,.95)),'classwise':{str(c):{'n':len(a:=[r['distance_to_gt'] for r in records if r['semantic_target_V17']==c]),'mean':float(np.mean(a)),'median':float(np.median(a))} for c in range(10)},'paired450_status':dict(collections.Counter(v['status'] for v in pairs)),'file_SHA256':sha(W/'proposals.npy'),'raw_array_SHA256':hashlib.sha256(p.tobytes()).hexdigest(),'typed_array_SHA256':mod.numpy_array_sha256(p),'hash_explanation':'Three distinct digest conventions, all consistent. No content mismatch.','upstream_outer_independence':'row-OOF true by recorded fold sets; mixed outerStage2 not nested. Same historicalfold cohort may isolate Stage1; existing BioMask still indirect contamination through Stage1proposal-trained otherfolds.'}
 dump(D/'audit.json',q);print(json.dumps(q,indent=2))
