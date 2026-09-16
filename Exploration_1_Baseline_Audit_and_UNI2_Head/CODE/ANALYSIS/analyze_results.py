import json,sys,csv
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent/'outputs'))
import numpy as np
from dataset import CLASSES

def mf(y,p):
    cm=np.bincount(10*y+p,minlength=100).reshape(10,10);den=cm.sum(0)+cm.sum(1)
    return np.divide(2*cm.diagonal(),den,out=np.zeros(10,dtype=float),where=den>0).mean()

def main():
    allrows=[]
    for family in ['cnn','uni2','lora']:
        directory=ROOT/('lora_results_v2' if family=='lora' else family+'_results')
        for p in sorted(directory.glob('*.json')):
            if p.name.endswith('.partial.json'):continue
            history=json.loads(p.read_text())
            if not isinstance(history,list) or not history or not isinstance(history[0],dict) or 'val' not in history[0]:continue
            best=max(history,key=lambda r:r['val']['macro_f1']);last=history[-1]
            row={'family':family,'recipe':p.stem,'epochs':len(history),'best_epoch':best['epoch'],
                 'best_macro_f1':best['val']['macro_f1'],'last_macro_f1':last['val']['macro_f1'],
                 'best_balanced_accuracy':best['val']['balanced_accuracy'],'best_ece':best['val']['ece15'],'best_nll':best['val']['nll'],
                 'final_train_macro':last['train']['macro_f1'],'best_f1':best['val']['f1'],'best_recall':best['val']['recall'],
                 'final_recall':last['val']['recall'],'support':best['val']['support'],
                 'objective_first':history[0]['objective'],'objective_last':last['objective'],
                 'objective_increases':sum(b['objective']>a['objective'] for a,b in zip(history,history[1:]))}
            allrows.append(row)
    (ROOT/'results_summary.json').write_text(json.dumps(allrows,indent=2))
    priors=[]
    rng=np.random.default_rng(17);seed17=np.load(ROOT/'uni2_results/ce_best.npz');rois=seed17['rois'];groups=np.unique(rois)
    pairs=[]
    for seed in [17,29,43]:
        suff='' if seed==17 else '_seed'+str(seed)
        a=np.load(ROOT/('uni2_results/la'+suff+'_best.npz'));b=np.load(ROOT/('uni2_results/ce'+suff+'_best.npz'))
        assert np.array_equal(a['labels'],b['labels']) and np.array_equal(a['rois'],b['rois'])
        pairs.append((a['labels'],a['logits'].argmax(1),b['logits'].argmax(1)))
    deltas=[]
    for _ in range(2000):
        sample=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([np.flatnonzero(rois==g) for g in sample])
        deltas.append(np.mean([mf(y[ix],a[ix])-mf(y[ix],b[ix]) for y,a,b in pairs]))
    stats={'mean_seed_paired_delta':float(np.mean([mf(y,a)-mf(y,b) for y,a,b in pairs])),
           'roi_bootstrap_percentile95':np.quantile(deltas,[.025,.975]).tolist(),'replicates':2000,
           'validation_rois':len(groups),'label':'exploratory: enriched sample and validation-selected checkpoints; not confirmatory'}
    (ROOT/'bootstrap.json').write_text(json.dumps(stats,indent=2))
    for family in ['cnn','uni2','lora']:
        print(family,[(r['recipe'],round(r['best_macro_f1'],4),r['best_epoch']) for r in allrows if r['family']==family])
    print(stats)

if __name__=='__main__':main()
