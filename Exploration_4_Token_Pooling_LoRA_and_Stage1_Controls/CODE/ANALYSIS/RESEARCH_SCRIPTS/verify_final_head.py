from bootstrap import *
import sys
sys.dont_write_bytecode=True
D=R/'EXPLORATION_4/ARCHITECTURES/FINAL_RECOMMENDED_A5';sys.path.insert(0,str(D/'CODE'))
import numpy as np,torch
from model import CachedClassifier
torch.set_num_threads(1);h=torch.nn.functional.layer_norm(torch.from_numpy(np.load(OLD/'02_cache/uni2_features/gt_fov96/features.npy')),(1536,));bio=np.load(OLD/'02_cache/biology/tierA.npy');out=[]
for seed in [17,29,43]:
 run=OLD/f'experiments/p3_confirm_A5_s{seed}';ck=torch.load(run/'model.pt',weights_only=True);n=ck['normalizer'];b=(bio-np.array(n['mean'],dtype=np.float32))/np.array(n['std'],dtype=np.float32);x=torch.cat([h,torch.from_numpy(b.astype('float32'))],1);m=CachedClassifier();m.load_state_dict(ck['model'],strict=True);saved=np.load(run/'predictions.npz')
 with torch.no_grad():z=m(x[saved['indices']]).numpy()
 err=float(np.abs(z-saved['logits']).max());assert np.allclose(z,saved['logits'],atol=2e-5,rtol=1e-5);out.append({'seed':seed,'max_error':err,'pass':True})
dump(D/'PROVENANCE/checkpoint_forward_verification.json',{'checks':out,'pass':True,'scope':'Final local CachedClassifier strictly loads all three historical A5 confirmation heads; forward versus saved logits'})
print(out)
