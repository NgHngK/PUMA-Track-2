import sys,time,json,collections,inspect
from pathlib import Path
from bootstrap import W,BASE,OLD,R,dump,sha,cp
sys.path[:0]=[str(BASE/'work/deps'),str(OLD/'01_shared_core')]
import torch,numpy as np,psutil
from PIL import Image
from dataset import read_manifest,centered_crop,image_tensor
from model import load_uni2,checkpoint_sha256
D=R/'EXPLORATION_4/TOKEN_AUDIT';C=D/'CACHE'
def main():
 torch.set_num_threads(8);torch.set_num_interop_threads(1);C.mkdir(exist_ok=True)
 rows=read_manifest(OLD/'01_sample_definition/sample_manifest.csv');n=len(rows);weights=Path('D:/Research/PUMA/Code/PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin')
 wh=checkpoint_sha256(weights);assert wh=='32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe'
 m=load_uni2(weights).eval().requires_grad_(False)
 source='\n\n'.join(inspect.getsource(getattr(type(m),x)) for x in ['_pos_embed','forward_features','forward_head'])
 (D/'timm_token_source.py.txt').write_text(source,encoding='utf8')
 done=np.load(C/'done.npy') if (C/'done.npy').exists() else np.zeros(n,dtype=bool)
 arr={}
 for name,shape in [('patch',(n,256,1536)),('prefix20',(n,265,1536)),('cls',(n,1536))]:
  p=C/(name+'.npy');arr[name]=np.lib.format.open_memmap(p,mode='r+' if p.exists() else 'w+',dtype='float32',shape=shape)
 ims=collections.OrderedDict();proc=psutil.Process();start=time.perf_counter();cpu=time.process_time();peak=0;captured={}
 handle=m.blocks[19].register_forward_hook(lambda mod,args,out:captured.update(prefix=out.detach()))
 def tensor(i):
  r=rows[i];p=r['image']
  if p not in ims:
   with Image.open(p) as im:ims[p]=im.convert('RGB')
   while len(ims)>2:ims.popitem(last=False)
  ims.move_to_end(p);return image_tensor(centered_crop(ims[p],r['x'],r['y'],96))
 ix=np.flatnonzero(~done)
 with torch.inference_mode():
  for k in range(0,len(ix),4):
   ids=ix[k:k+4];x=torch.stack([tensor(i) for i in ids]);z=m.forward_features(x)
   assert z.shape==(len(ids),265,1536) and m.num_prefix_tokens==9 and m.num_reg_tokens==8
   if k==0:
    embed=m.patch_embed(x);assert embed.shape[1:3]==(16,16)
    zz=captured['prefix'].clone()
    for block in m.blocks[20:]:zz=block(zz)
    zz=m.norm(zz)
    assert torch.equal(zz,z)
    old=np.load(OLD/'02_cache/uni2_features/gt_fov96/features.npy')[ids]
    err=float(np.max(np.abs(z[:,0].numpy()-old)))
    assert np.allclose(z[:,0].numpy(),old,rtol=1e-5,atol=2e-5)
    dump(D/'token_layout.json',{'tokens':265,'CLS_index':0,'register_indices':[1,8],'patch_indices':[9,264],'patch_grid':[16,16],'patch_size':14,'patch_order':'row-major x fastest; timm PatchEmbed NHWC then flatten','embedding':1536,'output_normalized_by_encoder_norm':True,'prefix20_shape':list(captured['prefix'].shape),'prefix_suffix_exact_parity':True,'cached_CLS_max_abs_error_first_batch':err,'timm_source':str(D/'timm_token_source.py.txt'),'GPU_available':torch.cuda.is_available()})
   assert torch.isfinite(z).all()
   arr['patch'][ids]=z[:,9:].numpy();arr['prefix20'][ids]=captured['prefix'].numpy();arr['cls'][ids]=z[:,0].numpy()
   for a in arr.values():a.flush()
   done[ids]=True;np.save(C/'done.npy',done);peak=max(peak,proc.memory_info().rss)
   if k%40==0:print('tokens',int(done.sum()),'/',n,'seconds',round(time.perf_counter()-start,1),flush=True)
 handle.remove()
 old=np.load(OLD/'02_cache/uni2_features/gt_fov96/features.npy');err=float(np.max(np.abs(arr['cls']-old)));assert np.allclose(arr['cls'],old,rtol=1e-5,atol=2e-5)
 dump(C/'contract.json',{'UIDs':[r['uid'] for r in rows],'coordinate_source':'GT area-centroid','FOV':96,'manifest_sha256':sha(OLD/'01_sample_definition/sample_manifest.csv'),'preprocessing_sha256':sha(OLD/'01_shared_core/dataset.py'),'checkpoint_sha256':wh,'token_schema':json.loads((D/'token_layout.json').read_text()),'files':{k:sha(C/(k+'.npy')) for k in arr},'images':{p:sha(Path(p)) for p in sorted({r['image'] for r in rows})},'FP32':True})
 dump(C/'complete.json',{'rows':n,'new_rows':len(ix),'wall_seconds':time.perf_counter()-start,'cpu_seconds':time.process_time()-cpu,'peak_RSS':peak,'CLS_parity_max_error':err,'all_complete':bool(done.all())})
 print('TOKEN CACHE COMPLETE',flush=True)
if __name__=='__main__':main()
