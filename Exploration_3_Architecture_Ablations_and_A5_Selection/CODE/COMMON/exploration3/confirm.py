import json,sys
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from exploration3_train import run
selection=json.loads((R/'90_comparative_analysis/prompt3_selection.json').read_text());kind=selection['cv_selected']
assert kind=='A5','Only selected candidate may be confirmed'
for seed in [17,29,43]:run({'id':f'p3_confirm_{kind}_s{seed}','architecture':kind,'control':'real','seed':seed,'fold':-1,'epochs':10,'lr':.001,'weight_decay':.01,'sampler':'balanced','tau':0,'evaluation':'one post-selection confirmation on reused original150; not external'})
