import sys,json,time
from pathlib import Path
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from train import run,dump
def conf(id,**kw):return dict(id=id,epochs=10,seed=17,fov=96,tau=1,sampler='natural',lr=.001,weight_decay=.01,**kw)
def state(key,value):
    p=R/'00_reference/RESEARCH_STATE.json';s=json.loads(p.read_text());s[key]=value;dump(p,s)
def execute(c):
    result=run(c);state('last_completed_experiment',c['id']);return result
def score(s):return s['selected']['val']['puma']['fixed10']['macro_f1']
def main():
    execute(conf('exp_biology_only',biology='tierA',biology_only=True,hypothesis='Does direct RGB biology generalize by itself?'))
    while not all((R/f'02_cache/uni2_features/gt_fov{f}/runtime.jsonl').exists() for f in [64,96,128]):time.sleep(10)
    state('phase','cached FOV and biology controlled experiments')
    fovs=[]
    for f in [64,96,128]:
        c=conf('exp_fov'+str(f)+'_la',hypothesis='Matched single-field-of-view screen');c['fov']=f;fovs.append(execute(c))
    best=max(fovs,key=score);default=next(s for s in fovs if s['config']['fov']==96)
    chosen=default if score(best)-score(default)<.002 else best;fov=chosen['config']['fov'];state('selected_fov',fov)
    losses=[chosen]
    for name,tau,sampler in [('ce',0,'natural'),('balanced_ce',0,'balanced')]:
        c=conf('exp_loss_'+name,hypothesis='Isolate effective prior and exposure');c.update(fov=fov,tau=tau,sampler=sampler);losses.append(execute(c))
    winner=max(losses,key=score);ce=next(s for s in losses if s['config']['tau']==0 and s['config']['sampler']=='natural')
    if score(winner)-score(ce)<.002:winner=ce
    recipe={'fov':fov,'tau':winner['config']['tau'],'sampler':winner['config']['sampler']};state('selected_recipe',recipe)
    base=winner;bio_results={}
    for bio in ['tierA','placebo','shuffle','oracle']:
        c=conf('exp_fusion_'+bio,biology=bio,hypothesis='Incremental biology with matched capacity and information controls');c.update(recipe);bio_results[bio]=execute(c)
    if score(bio_results['tierA'])>score(base):
        for family in ['stain','gradient_texture','ring','roi_relative']:
            c=conf('exp_family_'+family,biology='tierA',families=[family],hypothesis='Which biological family contributes beyond appearance?');c.update(recipe);execute(c)
        for seed in [29,43]:
            for bio in ['none','tierA','placebo','shuffle']:
                c=conf('exp_rep_'+bio+'_'+str(seed),biology=bio,hypothesis='Repeat apparent biological increment across initialization and sampling seeds');c.update(recipe);c['seed']=seed;execute(c)
    state('phase','screens complete; inspect comparisons before final architecture and hyperparameter refinement')
    print('PLANNED SCREENS COMPLETE',flush=True)
if __name__=='__main__':main()
