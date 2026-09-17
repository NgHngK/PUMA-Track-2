import json,csv,hashlib,zipfile,re,shutil
from pathlib import Path
import numpy as np
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';A=R/'00_reference/biomask_audit';O=R/'90_comparative_analysis'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def csvout(p,rows):
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
split=list(csv.DictReader((R/'01_sample_definition/prompt3_cv.csv').open()));cks={int(r['fold']):r for r in csv.DictReader((A/'checkpoint_index.csv').open())};rp=[]
for row in split:
    f=int(row['historical_biomask_fold']);rp.append({**row,'checkpoint_sha256':cks[f]['sha256'],'checkpoint_training_folds':cks[f]['training_folds'],'coordinate_source':'GT area centroid','scope':'row-wise OOF but not outer-independent'})
csvout(A/'row_provenance.csv',rp)
# Export every fit's selected per-class metrics without changing experimental source or outputs.
classes=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis'];per=[];epoch=[];fits=list((R/'experiments').glob('p3_*/summary.json'));assert len(fits)==237
for p in fits:
    s=json.loads(p.read_text());m=s['selected']['val'];cm=np.array(m['confusion'])
    for i,c in enumerate(classes):
        pm=m['puma']['summed']['class_metrics']['nuclei_'+c];fm=m['puma']['fixed10'];per.append({'id':s['id'],'class':c,'support':m['support'][i],'semantic_P':m['precision'][i],'semantic_R':m['recall'][i],'semantic_F1':m['f1'][i],'one_vs_rest_accuracy':float((cm.sum()-cm[i].sum()-cm[:,i].sum()+2*cm[i,i])/cm.sum()),'global_accuracy':m['accuracy'],'V17_TP':pm['TP'],'V17_FP':pm['FP'],'V17_FN':pm['FN'],'pooled_P':pm['precision'],'pooled_R':pm['recall'],'pooled_F1':pm['f1_score'],'ROI_P':fm['precision_by_class']['nuclei_'+c],'ROI_R':fm['recall_by_class']['nuclei_'+c],'ROI_F1':fm['f1_by_class']['nuclei_'+c]})
    hh=[json.loads(l) for l in (p.parent/'history.jsonl').read_text().splitlines()];assert len(hh)==10
    for h in hh:epoch.append({'id':s['id'],'epoch':h['epoch'],'objective':h['objective'],'train_CE':h['train_CE'],'val_CE':h['val_CE'],'train_semantic_F1':h['train']['macro_f1'],'val_semantic_F1':h['val']['macro_f1'],'ROI_F1':h['val']['puma']['fixed10']['macro_f1'],'pooled_F1':h['val']['puma']['summed']['macro_f1'],'accuracy':h['val']['accuracy'],'balanced_accuracy':h['val']['balanced_accuracy'],'gap':h['gap'],'min_class_exposure':min(h['exposure']),'min_positive_target_head_gradient':min(h['positive_target_head_gradient']),'LR':h['lr']})
csvout(O/'architecture_per_fit_per_class.csv',per);csvout(O/'architecture_epoch_dynamics.csv',epoch)
registry=list(csv.DictReader((R/'experiments/registry.csv').open()));ids=[r['experiment_id'] for r in registry];assert len(set(ids))==len(ids);assert len(registry)==267
validation=json.loads((R/'93_final_architecture/VERIFICATION.json').read_text());validation.update(prompt3_completed_fits=237,prompt3_completed_epochs=2370,registry_unique_entries=len(registry),mask_quality_n=450)
text=(R/'STAGE2_RESEARCH_REPORT.md').read_text(encoding='utf-8');links=re.findall(r'\]\(([^)]+)\)',text);bad=[l for l in links if ':' not in l and not l.startswith('#') and not (R/l).exists()];assert not bad,bad;validation['report_links']='PASS'
validation['source_and_mask_note']='TierB uses predicted masks only; GT mask directory is distinct. Inference checkpoints hash-matched, but all BioMask current outer comparisons are contaminated diagnostics.'
(R/'00_reference/PROMPT3_FINAL_AUDIT.json').write_text(json.dumps(validation,indent=2))
for p in W.glob('*.py'):shutil.copy2(p,R/'00_reference/prompt3_reproduction'/p.name)
csvout(A/'artifact_index.csv',[{'path':str(p.relative_to(R)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in A.rglob('*') if p.is_file() and p.name!='artifact_index.csv' and '__pycache__' not in p.parts])
for name in ['findings.md','research-log.md']:
    p=W.parent/name;t=p.read_text(encoding='utf-8');marker='Exploration3 completed (2026-09-08)'
    if marker not in t:p.write_text(t+'\n\n'+marker+': 234 finite groupedCV fits and3 fixed-final-epoch A5 confirmation fits. A5 rank8 selected (27866 trainable parameters); CV delta+.005997, reused150-cohort delta+.018823. Five BioMask finals hash-matched and450 masks measured; historical outer contamination blocks TierB promotion. Integrated report updated with explicit prior/new conclusions and confirmation intermediate-evaluation logging deviation. Stage1 unchanged; next is one full Stage1-population experiment, no further architecture exploration.\n',encoding='utf-8')
p=W.parent/'research-state.yaml';t=p.read_text();t+='\nprompt3_status: complete\nprompt3_selected_architecture: A5_rank8_UNI2_TierA_interaction\nprompt3_fits: 237\n' if 'prompt3_status:' not in t else '';p.write_text(t)
state=json.loads((R/'00_reference/RESEARCH_STATE.json').read_text());state['exploration3']['final_audit']='PROMPT3_FINAL_AUDIT.json';state['final_verification']=validation;state['next']=['One full fixed-Stage1 proposal population experiment with A5; verified provenance, natural calibration and locked test'];(R/'00_reference/RESEARCH_STATE.json').write_text(json.dumps(state,indent=2))
def files():return sorted(p for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='ARTIFACT_SHA256.csv')
csvout(R/'ARTIFACT_SHA256.csv',[{'path':p.relative_to(R).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in files()])
dest=R.parent/'PUMA_STAGE2_PROMPT3_BUNDLE.zip'
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files()+[R/'ARTIFACT_SHA256.csv']:z.write(p,Path(R.name)/p.relative_to(R))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None;members=len(z.namelist());manifest=list(csv.DictReader(z.read(R.name+'/ARTIFACT_SHA256.csv').decode().splitlines()))
    for r in manifest:assert hashlib.sha256(z.read(R.name+'/'+r['path'])).hexdigest()==r['sha256']
verify={'archive':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size,'members':members,'CRC_and_all_member_SHA256':'PASS','selected':'A5','prompt3_fits':237,'full_data_validation':False};dest.with_suffix('.verification.json').write_text(json.dumps(verify,indent=2));print(json.dumps(verify))
