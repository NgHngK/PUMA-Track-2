import ast,csv,hashlib,json,shutil,sys,datetime
from pathlib import Path
W=Path(__file__).parent;BASE=W.parent;R=BASE/'outputs/STAGE2_RESEARCH_20260907'
SRC=Path(r'D:\Research\PUMA\Code\Version 17\PUMA_Nuclei_Pipeline\src\puma_nuclei')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2),encoding='utf-8')
def main():
    ref=R/'00_reference';ref.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(r'C:\Users\Hngk\Downloads\prompt.md',ref/'MASTER_PROMPT.md')
    shutil.copyfile(BASE/'outputs/STAGE2_RESEARCH_REPORT.md',ref/'PRIOR_RESEARCH_REPORT.md')
    core=R/'01_shared_core';pkg=core/'puma_v17_evaluator';pkg.mkdir(parents=True,exist_ok=True)
    (pkg/'__init__.py').write_text('"""Vendored V17 evaluator; source modules remain byte-identical."""\n')
    selected=['constants.py','evaluation/__init__.py','evaluation/public_matcher.py','evaluation/aggregators.py','evaluation/dataset_evaluation.py','data/annotations.py']
    hashes={}
    for rel in selected:
        p=pkg/rel;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(SRC/rel,p);hashes[rel]={'source':str(SRC/rel),'sha256':sha(p)}
    (pkg/'data/__init__.py').write_text('')
    write(ref/'v17_evaluation_contract/source_hashes.json',hashes)
    for name in ['dataset.py','model.py','loss.py','train_eval.py']:
        shutil.copyfile(BASE/'outputs'/name,core/name)
    index=[]
    for p in SRC.rglob('*.py'):
        text=p.read_text(encoding='utf-8');tree=ast.parse(text)
        index.append({'path':str(p),'sha256':sha(p),'lines':len(text.splitlines()),'definitions':[{'name':n.name,'line':n.lineno} for n in ast.walk(tree) if isinstance(n,(ast.FunctionDef,ast.ClassDef))]})
    write(ref/'CODE_INDEX.json',index)
    old=json.loads((W/'results_summary.json').read_text())
    with (ref/'EXPERIMENT_INDEX.csv').open('w',newline='') as f:
        keys=['family','recipe','epochs','best_epoch','best_macro_f1','last_macro_f1'];w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(old)
    write(ref/'RESEARCH_STATE.json',{'status':'active','phase':'V17 parity and historical audit','stage1':'immutable coordinates-only; no tissue segmentation',
        'class_map':['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis'],
        'checkpoint':r'D:\Research\PUMA\Code\PUMA_pretrained_checkpoints\UNI2-h\uni2_h_model.bin','checkpoint_sha256':'32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe',
        'baseline':'single crop -> frozen UNI2 CLS -> nonaffine LN -> Linear10','census_n':97193,'rois':205,
        'split':'preserve existing ROI-grouped development split; patient mapping unverified','prior_runs':old,
        'new_findings':['V17 internal IDs7/8 differ; remap by name only at evaluation boundary','V17 GT centroid is serialized vertex mean; prior feature cache used area centroids'],
        'hypotheses':['FOV64/96/128','CE/LA/balanced','Tier A biology incremental value','oracle morphology diagnostic'],
        'next':['parity test','historical components index','prospective sample/runtime protocol'],
        'rejected_without_new_evidence':['tissue branch','GT neighbour semantics in deployable models','large fusion','new LoRA before simple controls']})
    shutil.copyfile(W/'ingestion/annotation_audit.json',ref/'dataset_census.json')
    print(R)
if __name__=='__main__':main()
