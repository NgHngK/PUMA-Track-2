import sys,json
from pathlib import Path
import numpy as np
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from train import run,dump
from dataset import read_manifest
def load(name):return json.loads((R/'experiments'/name/'metrics/summary.json').read_text())
def main():
    deltas=[];scores={}
    for seed in [17,29,43]:
        names={b:('exp_loss_balanced_ce' if b=='none' else 'exp_fusion_'+b) if seed==17 else 'exp_rep_'+b+'_'+str(seed) for b in ['none','tierA','placebo','shuffle']}
        ss={b:load(n) for b,n in names.items()};scores[seed]={b:s['selected']['val']['puma']['fixed10']['macro_f1'] for b,s in ss.items()}
        arr={}
        for b,s in ss.items():
            rr=s['selected']['val']['puma']['roi_metrics'];arr[b]=np.array([np.mean([r.get('nuclei_'+c,{}).get('f1_score',0.) for c in ['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']]) for r in rr])
        deltas.append(arr['tierA']-arr['none'])
    delta=np.mean(deltas,axis=0);rng=np.random.default_rng(17);boots=delta[rng.integers(0,len(delta),(2000,len(delta)))].mean(1);ci=np.percentile(boots,[2.5,97.5])
    eligible=all(s['tierA']>s['none'] for s in scores.values()) and all(np.mean([s['tierA']-s[b] for s in scores.values()])>0 for b in ['placebo','shuffle']) and ci[0]>0
    result={'seed_scores':scores,'mean_roi_delta':float(delta.mean()),'roi_bootstrap_95':ci.tolist(),'validation_rois':len(delta),'bootstrap_replicates':2000,'eligible_for_family_removal':bool(eligible),'caveat':'development-selected checkpoints and enriched fixed sample; exploratory, not external confirmation'}
    dump(R/'90_comparative_analysis/biology_promotion.json',result);print(result,flush=True)
    if eligible:
        families=['stain','gradient_texture','ring','roi_relative']
        for family in families:
            run(dict(id='exp_drop_'+family,epochs=10,seed=17,fov=96,tau=0,sampler='balanced',lr=.001,weight_decay=.01,biology='tierA',families=[f for f in families if f!=family],hypothesis='Leave one family out of the qualifying full TierA model'))
if __name__=='__main__':main()
