import sys,importlib.util,collections,json
sys.dont_write_bytecode=True
from bootstrap import *
import numpy as np,torch
from engine import load_data as original_data,norm,P,extra_metrics,Model
def prepare():
 rows,cv,y=original_data();pairs=json.loads((P/'STAGE1_PROPOSALS/paired450.json').read_text());selected=[v['index'] for v in pairs if v['status']=='matched' and v['historical_fold']==0];rr=[rows[i].copy() for i in selected];yy=y[selected];cc=cv[selected].copy();cc[cc>=0]=0
 cache=P/'STAGE1_PROPOSALS/CACHE';srows=json.loads((cache/'manifest.json').read_text());mapping={int(k):i for i,k in enumerate(np.load(cache/'original_indices.npy'))};si=np.array([mapping[i] for i in selected]);stage=[]
 for i,j in zip(selected,si):
  r=rows[i].copy();r.update(x=srows[j]['x'],y=srows[j]['y'],coordinate_source='stage1_oof');stage.append(r)
 hgt=norm(np.load(P/'TARGET_POOLING/REPRESENTATIONS/A3.npy')[selected]);bg=np.load(OLD/'02_cache/biology/tierA.npy')[selected];bs=np.load(cache/'tierA.npy')[si];patch=np.load(cache/'patch.npy',mmap_mode='r');hs=[]
 for j,r in zip(si,stage):
  cx,cy=np.floor([(r[k]-np.floor(r[k]-48+.5))/6 for k in ['x','y']]).astype(int);near=[yy*16+xx for yy in range(cy-1,cy+2) for xx in range(cx-1,cx+2)];hs.append(patch[j,near].mean(0))
 hs=norm(np.array(hs));dump(Path(__file__).resolve().parents[1]/'PROVENANCE/new_clean_cohort.json',{'original_indices':selected,'stage1_cache_indices':si.tolist(),'train':int((cc>=0).sum()),'val':int((cc<0).sum()),'train_counts':np.bincount(yy[cc>=0],minlength=10).tolist(),'val_counts':np.bincount(yy[cc<0],minlength=10).tolist(),'excluded_detector_training_fold':0,'source_checkpoint_hash':sha(P/'STAGE1_PROPOSALS/stage1_final_fold0.pt')});return rr,stage,cc,yy,hgt,hs,bg,bs
