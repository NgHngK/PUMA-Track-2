import sys,time,json,collections
sys.dont_write_bytecode=True
from bootstrap import *
sys.path[:0]=[str(BASE/'work/deps'),str(OLD/'01_shared_core')]
import numpy as np,torch,psutil
from PIL import Image
from dataset import read_manifest,centered_crop,image_tensor
from model import load_uni2
from biology import roi_channels,point_features
D=R/'EXPLORATION_4/STAGE1_PROPOSALS';C=D/'CACHE'
if __name__=='__main__':
 torch.set_num_threads(8);torch.set_num_interop_threads(1);C.mkdir(exist_ok=True)
 if (C/'complete.json').exists():print('Existing complete cache');sys.exit()
 original=read_manifest(OLD/'01_sample_definition/sample_manifest.csv');pairs=json.loads((D/'paired450.json').read_text());rows=[];indices=[]
 for pair in pairs:
  if pair['status']=='matched':
   r=original[pair['index']].copy();r.update(x=pair['stage1_x'],y=pair['stage1_y'],coordinate_source='stage1_oof',proposal_uid=pair['proposal_uid']);rows.append(r);indices.append(pair['index'])
 n=len(rows);dump(C/'manifest.json',rows);np.save(C/'original_indices.npy',indices)
 done=np.load(C/'done.npy') if (C/'done.npy').exists() else np.zeros(n,bool);arr={}
 for key,shape in [('cls',(n,1536)),('patch',(n,256,1536)),('tierA',(n,16))]:
  path=C/(key+'.npy');arr[key]=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',shape=shape,dtype='float32')
 model=load_uni2('D:/Research/PUMA/Code/PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin').eval().requires_grad_(False);cache=collections.OrderedDict();t=time.perf_counter();cpu=time.process_time();peak=0;proc=psutil.Process()
 def inp(i):
  r=rows[i];path=r['image']
  if path not in cache:
   with Image.open(path) as im:rgb=im.convert('RGB');ch=roi_channels(np.array(rgb));cache[path]=(rgb,ch)
   while len(cache)>2:cache.popitem(last=False)
  cache.move_to_end(path);rgb,ch=cache[path];arr['tierA'][i]=point_features(ch,r['x'],r['y']);return image_tensor(centered_crop(rgb,r['x'],r['y'],96))
 ids=np.flatnonzero(~done)
 with torch.inference_mode():
  for k in range(0,len(ids),4):
   ix=ids[k:k+4];z=model.forward_features(torch.stack([inp(i) for i in ix]));assert torch.isfinite(z).all() and z.shape[1:]==(265,1536);arr['cls'][ix]=z[:,0].numpy();arr['patch'][ix]=z[:,9:].numpy()
   for v in arr.values():v.flush()
   done[ix]=True;np.save(C/'done.npy',done);peak=max(peak,proc.memory_info().rss)
   if k%40==0:print('proposal tokens',done.sum(),'/',n,'seconds',round(time.perf_counter()-t,1),flush=True)
 dump(C/'complete.json',{'n':n,'wall_seconds':time.perf_counter()-t,'cpu_seconds':time.process_time()-cpu,'peak_RSS':peak,'hashes':{key:sha(C/(key+'.npy')) for key in arr},'proposal_sha256':sha(D/'proposals.npy'),'source':sha(Path(__file__)),'coordinate_source':'fixed Stage1 matched proposals; no changes','UIDs':[r['uid'] for r in rows],'checkpoint_sha256':'32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe'})
 print('PROPOSAL TOKEN CACHE COMPLETE',flush=True)
