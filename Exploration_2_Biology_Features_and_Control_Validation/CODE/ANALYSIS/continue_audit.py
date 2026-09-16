import sys,json,csv,hashlib,ast,collections,zipfile
from pathlib import Path
import numpy as np
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from dataset import read_manifest,CLASSES
from metrics import puma_metrics
SRC=Path(r'D:\Research\PUMA\Code\Version 17\PUMA_Nuclei_Pipeline\src\puma_nuclei')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    details={
    'biology_features.py':('predicted mask probability,H proxy,gradient,quality,ROI median/IQR','Tier B only with inference-safe OOF masks; GT-mask substitution is Tier C','No class argument; mask provenance controls deployability','Descriptive biology association and combined historical run; no isolated matched increment proven'),
    'fusion.py':('appearance/biology/reliability/context embeddings','Inputs conditional on provenance','Dense encoders and zero-initialized bounded residuals/gates; not inherently leakage','Large-capacity matched controls not established by historical aggregate curve'),
    'biomask.py':('RGB and point prompt','Tier B architecture can run without GT polygons','Training supervision uses masks; inference must use network outputs','No validated checkpoint with outer-development-group exclusion recovered'),
    'biomask_data.py':('RGB,proposal,GT polygon training target','Training-only targets must remain separate','Rasterizes polygons for supervision; target use is not itself inference leakage','OOF by target fold does not establish exclusion of all outer validation groups'),
    'biomask_sampling.py':('positive/negative targets and proposal strata','Training-only sampler','GT-dependent sampling is supervised training, not deployable descriptor','No controlled sampler-only attribution'),
    'biomask_trainer.py':('fold training targets,network','Requires audited checkpoint provenance','Held-fold exclusion code exists; verify concrete folds before use','No new mask training justified before Tier A incremental test'),
    'model.py':('five identity pools,biology,reliability,local/tissue/spatial','Mixed deployability; image tissue-context differs from tissue segmentation but outside minimal candidate','Many residual/gated pathways; entropy and margin detached; original checkpoint online UNI2','Historical early semantic improvement then increasing validation loss; no causal proof of biology starvation'),
    'dataset.py':('RGB crops,cached handcrafted/token maps,targets','Input/target dataclasses separated','Identity token maps are mask weights, not cached UNI2 CLS vectors','Avoid silently describing old outputs as offline foundation features'),
    'preprocess.py':('Stage1 OOF proposals,BioMask checkpoints,GT supervision,ROI images','Conditional on fold provenance','Checks held fold not trained; builds predicted reliability and cached biology','Cache/source signatures exist; outer-fold leakage contract must be checked separately'),
    'trainer.py':('model,bank,semantic/utility targets','Training and deployment code mixed','Phased feature/adapter optimization; semantic probability times utility score','Recovered 18 epochs: LoRA gradient active from epoch5; validation loss diverges; multiple checkpoint estimands'),
    'sampling.py':('GT resolved groups,quality/mask/offset predictions','Supervised training only','Anchor plus diverse proposal views, effective sample accounting','Repeated localization views not independent biological cases'),
    'encoder.py':('RGB64/128/256,verifiedUNI2','Inference available','CLS/center/ring pooling; actual online encoder gradients when enabled','The minimal CLS baseline has a different pooling contract'),
    'lora.py':('fused qkv linear modules','Inference available','Historical fused QKV residual can update K too; new restricted Q/V is different','Positive adapter gradient in old log does not prove all classes generalize')}
    rows=[]
    for name,(inputs,deploy,gt,prior) in details.items():
        p=SRC/'stage2'/name;rows.append({'component':name,'path':str(p),'sha256':sha(p),'inputs':inputs,'deployability':deploy,'GT_or_gradient_notes':gt,'prior_test_and_fairness':prior,'held_out_evidence':'combined historical run; no isolated component attribution','decision':'audit; minimal controlled ablation first'})
    ref=R/'00_reference'
    with (ref/'HISTORICAL_COMPONENT_INDEX.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    remote={}
    for name in ['biology_features.py','fusion.py']:
        p=ref/'historical_code_audit'/('drive_'+name);local=SRC/'stage2'/name
        remote[name]={'remote_saved_sha256':sha(p),'local_sha256':sha(local),'AST_equal':ast.dump(ast.parse(p.read_text()))==ast.dump(ast.parse(local.read_text()))}
    (ref/'historical_code_audit/drive_comparison.json').write_text(json.dumps(remote,indent=2))
    census=collections.Counter();components=collections.Counter();multis=[]
    ann=Path(r'D:\Research\PUMA\Code\TRAINING CODE\Dataset\01_training_dataset_geojson_nuclei')
    for p in ann.glob('*.geojson'):
        for i,f in enumerate(json.loads(p.read_text())['features']):
            label=f['properties']['classification']['name'];g=f['geometry'];n=1 if g['type']=='Polygon' else len(g['coordinates']);census[label]+=1;components[label]+=n
            if n>1:multis.append({'file':p.name,'feature':i,'class':label,'components':n})
    (ref/'dataset_audit').mkdir(exist_ok=True);(ref/'dataset_audit/component_census.json').write_text(json.dumps({'semantic_features':dict(census),'V17_exterior_components':dict(components),'MultiPolygons':multis},indent=2))
    sample=read_manifest(R/'01_sample_definition/sample_manifest.csv');val=[r for r in sample if r['split']=='val'];labels=np.array([r['label'] for r in val]);ceiling=puma_metrics(val,labels,np.eye(10)[labels],trace=False)
    (ref/'dataset_audit/perfect_semantic_control.json').write_text(json.dumps(ceiling,indent=2))
    print('AST comparisons',remote);print('features/components',sum(census.values()),sum(components.values()),'MultiPolygon',len(multis));print('perfect semantic ROI/pooled',ceiling['fixed10']['macro_f1'],ceiling['summed']['macro_f1'])
if __name__=='__main__':main()
