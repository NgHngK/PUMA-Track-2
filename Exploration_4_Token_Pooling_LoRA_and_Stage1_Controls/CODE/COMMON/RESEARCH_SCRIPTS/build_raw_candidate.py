"""Prepare raw-image A3 implementation; recommendation awaits final study decision."""
from bootstrap import *
src=OLD/'93_final_architecture';dest=W/'raw_a3_candidate';dest.mkdir(exist_ok=True)
for p in src.rglob('*'):
 if p.is_file() and '__pycache__' not in p.parts:cp(p,dest/p.relative_to(src))
pool='''import torch
from torch import nn

def patch_coordinates(points, fov=96):
    points=torch.as_tensor(points)
    return (points-torch.floor(points-fov/2+.5))/(fov/16)

def target_neighborhood(tokens, points, fov=96):
    if tokens.ndim!=3 or tokens.shape[1:]!=(265,1536):
        raise ValueError('Expected UNI2-h CLS + 8 registers + 256 patch tokens')
    if fov!=96:raise ValueError('Selected representation requires source FOV96')
    xy=patch_coordinates(torch.as_tensor(points,device=tokens.device,dtype=torch.float64),fov)
    if xy.shape!=(len(tokens),2):raise ValueError('Supply one original x,y point per crop')
    centers=xy.floor().long().clamp(0,15)
    grid=torch.arange(256,device=tokens.device)
    near=((grid[None]%16-centers[:,0,None]).abs()<=1)&((grid[None]//16-centers[:,1,None]).abs()<=1)
    # Index then mean preserves the executed nine-token reduction order.
    return torch.stack([tokens[i,9:][near[i]].mean(0) for i in range(len(tokens))])
'''
(dest/'pooling.py').write_text(pool)
text=(src/'model.py').read_text().replace('Selected A5 rank-eight appearance x TierA interaction; no BioMask or LoRA.','A3 target representation with the retained rank-eight appearance x TierA head.').replace('LN CLS1536','LN target1536').replace('def forward(self,images,normalized_biology):','def forward(self,images,normalized_biology,points):').replace('with torch.no_grad():h=nn.functional.layer_norm(self.encoder(images).float(),(1536,))','from pooling import target_neighborhood\n        with torch.no_grad():h=nn.functional.layer_norm(target_neighborhood(self.encoder.forward_features(images).float(),points),(1536,))')
(dest/'model.py').write_text(text)
text=(src/'features.py').read_text().replace('from uni2 import load_uni2,checkpoint_sha256','from uni2 import load_uni2,checkpoint_sha256\nfrom pooling import target_neighborhood').replace("'checkpoint_sha256':config['checkpoint_sha256']}","'checkpoint_sha256':config['checkpoint_sha256'],'representation':config['representation'],'token_layout':'CLS0-registers1:9-patches9:265-rowmajor16x16'}").replace("['features.py','dataset.py','crop.py','uni2.py']","['features.py','dataset.py','crop.py','uni2.py','pooling.py']").replace('z=m(x).float().cpu().numpy()',"z=target_neighborhood(m.forward_features(x).float(),[[rows[i]['x'],rows[i]['y']] for i in ii],config['fov']).cpu().numpy()")
(dest/'features.py').write_text(text)
text=(src/'train_eval.py').read_text().replace("config.get('architecture')!='A5'","config.get('architecture')!='P4_A3'")
(dest/'train_eval.py').write_text(text)
cfg=json.loads((src/'config.json').read_text());cfg.update(architecture='P4_A3',representation='target_neighborhood_3x3',epochs=20,patience=5);dump(dest/'config.json',cfg)
dump(dest/'SOURCE_LINEAGE.json',{'original_package':str(src),'original_hashes':{str(p.relative_to(src)):sha(p) for p in src.rglob('*.py') if '__pycache__' not in p.parts},'new_modules':['pooling.py'],'modified_modules':['model.py','features.py','train_eval.py'],'status':'candidate implementation; recommendation awaits F and final confirmation'})
print(dest)
