import sys,json
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from exploration3_train import run
families=[('A0','real'),('A1','real')]+[(f'A{i}',c) for i in range(2,7) for c in ['real','placebo','shuffle']]+[(f'B{i}','real') for i in range(5)]+[(b,c) for b in ['B3','B4'] for c in ['placebo','shuffle']]
if __name__=='__main__':
    for kind,control in families:
        for seed in [17,29,43]:
            for fold in range(3):run({'id':f'p3_{kind}_{control}_s{seed}_cv{fold}','architecture':kind,'control':control,'seed':seed,'fold':fold,'epochs':10,'lr':.001,'weight_decay':.01,'sampler':'balanced','tau':0})
