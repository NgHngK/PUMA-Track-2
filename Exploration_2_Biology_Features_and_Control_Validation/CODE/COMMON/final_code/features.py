"""One deterministic frozen UNI2 extraction pass with content-addressed provenance."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import numpy as np,torch
from PIL import Image
from dataset import read_manifest
from crop import centered_crop,image_tensor
from uni2 import load_uni2,checkpoint_sha256
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def input_contract(config):return {'fov':config['fov'],'biology':bool(config.get('biology_families')),'checkpoint_sha256':config['checkpoint_sha256']}
def extract(manifest,weights,config,out,device='cpu',batch=4,threads=8):
    torch.set_num_threads(threads);rows=read_manifest(manifest);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    images={r['image'] for r in rows};expected=config['checkpoint_sha256'];actual=checkpoint_sha256(weights)
    if actual!=expected:raise ValueError('Wrong UNI2 checkpoint SHA256')
    contract={'manifest_sha256':sha(manifest),'checkpoint_sha256':actual,'input_contract':input_contract(config),'image_hashes':{p:sha(p) for p in sorted(images)},
              'source_hashes':{n:sha(Path(__file__).parent/n) for n in ['features.py','dataset.py','crop.py','uni2.py']+(['biology.py'] if config.get('biology_families') else [])},'precision':'FP32','n':len(rows)}
    cp=out/'contract.json'
    if cp.exists():
        if json.loads(cp.read_text())!=contract:raise ValueError('Cache contract mismatch')
        if (out/'complete.json').exists():return out
        raise FileExistsError('Incomplete cache preserved; use a new cache directory')
    cp.write_text(json.dumps(contract,indent=2))
    m=load_uni2(weights).to(device).eval().requires_grad_(False)
    features=np.lib.format.open_memmap(out/'features.npy',mode='w+',dtype='float32',shape=(len(rows),1536))
    use_bio=bool(config.get('biology_families'));bio=np.zeros((len(rows),16),dtype=np.float32) if use_bio else None
    if use_bio:from biology import roi_channels,point_features
    groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[r['image']].append(i)
    start=time.perf_counter();seen=0
    with torch.inference_mode():
        for path,ix in groups.items():
            with Image.open(path) as im:rgb=im.convert('RGB')
            if use_bio:
                ch=roi_channels(np.asarray(rgb))
                for i in ix:bio[i]=point_features(ch,rows[i]['x'],rows[i]['y'])
            for k in range(0,len(ix),batch):
                ii=ix[k:k+batch];x=torch.stack([image_tensor(centered_crop(rgb,rows[i]['x'],rows[i]['y'],config['fov'])) for i in ii]).to(device)
                z=m(x).float().cpu().numpy()
                if not np.isfinite(z).all():raise ValueError('Nonfinite UNI2 features')
                features[ii]=z;seen+=len(ii)
            features.flush();print(f'Cached {seen}/{len(rows)} nuclei',flush=True)
    if bio is not None:np.save(out/'biology.npy',bio)
    (out/'complete.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'feature_sha256':sha(out/'features.npy'),'n':seen},indent=2));return out
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--weights',required=True);p.add_argument('--config',default='config.json');p.add_argument('--out',required=True);p.add_argument('--device',default='cpu');p.add_argument('--batch',type=int,default=4);p.add_argument('--threads',type=int,default=8);a=p.parse_args()
    extract(a.manifest,a.weights,json.loads(Path(a.config).read_text()),a.out,a.device,a.batch,a.threads)
