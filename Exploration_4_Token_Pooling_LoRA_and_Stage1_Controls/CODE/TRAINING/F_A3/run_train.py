import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import engine
p=argparse.ArgumentParser(description="Run this frozen representation with local code")
p.add_argument('--config',required=True);p.add_argument('--output',required=True)
a=p.parse_args();cfg=json.loads(Path(a.config).read_text());cfg['output_root']=str(Path(a.output).resolve())
import numpy as np,torch
contract=json.loads((engine.P/'BIOMASK_GUIDED/REPRESENTATIONS/contract.json').read_text());ix=np.array(contract['original_indices']);rr,cc,yy=engine.load_data();rows=[rr[i] for i in ix];cv=cc[ix].copy();cv[cv>=0]=0;y=yy[ix];bio=np.load(engine.OLD/'02_cache/biology/tierA.npy')[ix];ti=np.flatnonzero(cv>=0);mean=bio[ti].mean(0);std=np.maximum(bio[ti].std(0),1e-6);h=torch.from_numpy(np.load(engine.P/'BIOMASK_GUIDED/REPRESENTATIONS'/(cfg['family']+'.npy')));x=torch.cat((h,torch.from_numpy(((bio-mean)/std).astype('float32'))),1);engine.load_data=lambda:(rows,cv,y);engine.inputs=lambda cfg,ti:(x,{'mean':mean.tolist(),'std':std.tolist()})
engine.run(cfg)
