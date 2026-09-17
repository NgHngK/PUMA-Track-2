"""Finite, prospectively registered clean-fold mask pooling study."""
import json
import engine as e
from bootstrap import *
import numpy as np,torch
P=e.P
def prepare():
 rows,cv,y=e.load_data();D=P/'BIOMASK_GUIDED/CLEAN_GT_ONLY'
 assert (D/'complete.json').exists(),'Clean mask training must finish first'
 rm=np.load(OLD/'00_reference/biomask_audit/roi_manifest.npy');fold=np.load(OLD/'00_reference/biomask_audit/folds.npy');fm=dict(zip(rm['roi_id'],fold));ix=np.array([i for i,r in enumerate(rows) if fm[r['roi']]==0]);cc=cv[ix].copy();cc[cc>=0]=0
 masks=np.load(D/'masks.npy');origins=np.load(D/'transforms.npy');patch=np.load(P/'TOKEN_AUDIT/CACHE/patch.npy',mmap_mode='r');rng=np.random.default_rng(1701);shuffle=ix.copy()
 for part in [np.flatnonzero(cc>=0),np.flatnonzero(cc<0)]:shuffle[part]=rng.permutation(ix[part])
 geom=np.load(P/'TARGET_POOLING/REPRESENTATIONS/pooling_geometry.npz');pooled=[];broken=[];records=[];ww=[];wb=[]
 for i,j in zip(ix,shuffle):
  left,top=[int(np.floor(rows[i][k]-48+.5)) for k in ['x','y']]
  for source,output,weights_out,label in [(i,pooled,ww,'real'),(j,broken,wb,'shuffled')]:
   # Shuffle the centered mask shape; translate only by the recipient crop convention.
   dx=int(origins[i,0]-left);dy=int(origins[i,1]-top);aligned=np.zeros((96,96),np.float32)
   x0=max(0,dx);x1=min(96,96+dx);y0=max(0,dy);y1=min(96,96+dy)
   if x1>x0 and y1>y0:aligned[y0:y1,x0:x1]=masks[source,y0-dy:y1-dy,x0-dx:x1-dx]
   w=aligned.reshape(16,6,16,6).mean((1,3)).reshape(256);mass=float(w.sum());fallback=mass<1e-12;w=geom['real'][i].copy() if fallback else w/mass
   output.append((patch[i]*w[:,None]).sum(0));weights_out.append(w);records.append({'index':int(i),'uid':rows[i]['uid'],'source_mask_index':int(source),'kind':label,'dx':dx,'dy':dy,'weight_mass':mass,'fallback':fallback})
 h=e.norm(np.array(pooled));b=e.norm(np.array(broken));cls=e.norm(np.load(OLD/'02_cache/uni2_features/gt_fov96/features.npy')[ix]);reps={'F_A3':e.norm(np.load(P/'TARGET_POOLING/REPRESENTATIONS/A3.npy')[ix]),'F_GAUSSIAN':e.norm(np.load(P/'TARGET_POOLING/REPRESENTATIONS/A2.npy')[ix]),'F_MASK':h,'F_CLS_MASK':e.norm(((cls+h)/2).numpy()),'F_SHUFFLE':b,'F_CLS_SHUFFLE':e.norm(((cls+b)/2).numpy())}
 dest=P/'BIOMASK_GUIDED/REPRESENTATIONS';dest.mkdir(exist_ok=True)
 for k,v in reps.items():
  path=dest/(k+'.npy')
  if path.exists():assert np.array_equal(np.load(path),v.numpy())
  else:np.save(path,v.numpy())
 if not (dest/'weights.npz').exists():np.savez_compressed(dest/'weights.npz',indices=ix,shuffle_indices=shuffle,real=np.array(ww),shuffled=np.array(wb))
 if not (dest/'alignment.json').exists():dump(dest/'alignment.json',records)
 dump(dest/'contract.json',{'original_indices':ix.tolist(),'train_n':int((cc>=0).sum()),'val_n':int((cc<0).sum()),'clean_checkpoint':sha(D/'final.pt'),'mask_sha256':sha(D/'masks.npy'),'protocol':sha(P/'PROVENANCE/F_IMPLEMENTATION_PROTOCOL.md'),'features':{k:sha(dest/(k+'.npy')) for k in reps}})
 return [rows[i] for i in ix],cc,y[ix],reps,np.load(OLD/'02_cache/biology/tierA.npy')[ix]
def main():
 rows,cv,y,reps,bio=prepare();e.load_data=lambda:(rows,cv,y)
 def inputs(cfg,ti):
  mean=bio[ti].mean(0);std=np.maximum(bio[ti].std(0),1e-6)
  return torch.cat([reps[cfg['family']],torch.from_numpy(((bio-mean)/std).astype('float32'))],1),{'mean':mean.tolist(),'std':std.tolist()}
 e.inputs=inputs
 for family in reps:
  for seed in [17,29,43]:e.run({'id':f'{family}_s{seed}','family':family,'seed':seed,'fold':-1,'epochs':10,'sampler':'balanced','population':'all GT nuclei historical fold0 original membership; clean GT-only mask','representation':family})
if __name__=='__main__':main()
