import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import engine
p=argparse.ArgumentParser(description="Run this frozen representation with local code")
p.add_argument('--config',required=True);p.add_argument('--output',required=True)
a=p.parse_args();cfg=json.loads(Path(a.config).read_text());cfg['output_root']=str(Path(a.output).resolve())
from alignment_data import prepare
import numpy as np,torch
gt,st,cv,y,hg,hs,bg,bs=prepare();ti=np.flatnonzero(cv>=0);is_stage=cfg['family']=='C_STAGE1TRAIN';rows=st if is_stage else gt;b=bs if is_stage else bg;h=hs if is_stage else hg;mean=b[ti].mean(0);std=np.maximum(b[ti].std(0),1e-6);x=torch.cat((h,torch.from_numpy(((b-mean)/std).astype('float32'))),1)
engine.load_data=lambda:(rows,cv,y)
engine.inputs=lambda cfg,ti:(x,{'mean':mean.tolist(),'std':std.tolist()})
engine.run(cfg)
