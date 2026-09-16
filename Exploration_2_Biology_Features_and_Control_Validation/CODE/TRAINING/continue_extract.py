import sys,json,time,hashlib,collections
from pathlib import Path
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path[:0]=[str(W/'deps'),str(R/'01_shared_core')]
import torch,numpy as np,psutil
from PIL import Image
from dataset import read_manifest,centered_crop,image_tensor
from model import load_uni2,checkpoint_sha256
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    torch.set_num_threads(8);torch.set_num_interop_threads(1)
    manifest=R/'01_sample_definition/sample_manifest.csv';rows=read_manifest(manifest)
    weights=Path(r'D:\Research\PUMA\Code\PUMA_pretrained_checkpoints\UNI2-h\uni2_h_model.bin');weight_hash=checkpoint_sha256(weights)
    assert weight_hash=='32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe'
    image_hashes={p:sha(Path(p)) for p in sorted({r['image'] for r in rows})}
    m=load_uni2(weights).eval().requires_grad_(False);proc=psutil.Process()
    image_cache=collections.OrderedDict()
    def tensor(i,fov):
        r=rows[i];path=r['image']
        if path not in image_cache:
            with Image.open(path) as im:image_cache[path]=im.convert('RGB')
            while len(image_cache)>2:image_cache.popitem(last=False)
        image_cache.move_to_end(path)
        return image_tensor(centered_crop(image_cache[path],r['x'],r['y'],fov))
    for fov in [96,64,128]:
        dest=R/'02_cache/uni2_features'/('gt_fov'+str(fov));dest.mkdir(parents=True,exist_ok=True)
        contract={'manifest_sha256':sha(manifest),'checkpoint_sha256':weight_hash,'image_sha256':image_hashes,
                  'transform_sha256':sha(R/'01_shared_core/dataset.py'),'encoder_sha256':sha(R/'01_shared_core/model.py'),
                  'fov':fov,'resize':224,'coordinate_source':'GT area-centroid control','mode':'frozen FP32 CLS','n':len(rows),'uids':[r['uid'] for r in rows]}
        contract_path=dest/'contract.json'
        if contract_path.exists():assert json.loads(contract_path.read_text())==contract,'Cache contract mismatch'
        else:contract_path.write_text(json.dumps(contract,indent=2))
        done_path=dest/'done.npy';done=np.load(done_path) if done_path.exists() else np.zeros(len(rows),dtype=bool)
        featpath=dest/'features.npy';features=np.lib.format.open_memmap(featpath,mode='r+' if featpath.exists() else 'w+',dtype='float32',shape=(len(rows),1536))
        start=time.perf_counter();cpu=time.process_time();peak=0;indices=np.flatnonzero(~done);nnew=len(indices)
        with torch.inference_mode():
            for k in range(0,len(indices),4):
                ix=indices[k:k+4];x=torch.stack([tensor(i,fov) for i in ix]);z=m(x).float()
                if not torch.isfinite(z).all():raise ValueError('Nonfinite features')
                features[ix]=z.numpy();done[ix]=True;features.flush();np.save(done_path,done);peak=max(peak,proc.memory_info().rss)
                if k%40==0:print('fov',fov,'complete',int(done.sum()),'/',len(rows),'sec',round(time.perf_counter()-start,1),flush=True)
        elapsed=time.perf_counter()-start
        run={'fov':fov,'new_examples':nnew,'seconds':elapsed,'cpu_seconds':time.process_time()-cpu,'examples_per_second':nnew/max(elapsed,1e-9),'peak_sampled_rss_gb':peak/1e9,'features_sha256':sha(featpath),'complete':bool(done.all()),'threads':8,'batch':4,'legacy_cache_reuse':False,'reason':'new manifest; legacy cache lacks image-content hashes; archived unchanged'}
        with (dest/'runtime.jsonl').open('a') as f:f.write(json.dumps(run)+'\n')
        state=R/'00_reference/RESEARCH_STATE.json';s=json.loads(state.read_text());s['phase']='feature extraction';s.setdefault('completed_caches',{})[str(fov)]=run;state.write_text(json.dumps(s,indent=2));print('DONE',run,flush=True)
if __name__=='__main__':main()
