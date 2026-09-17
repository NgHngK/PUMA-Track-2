import sys,json
sys.dont_write_bytecode=True
from bootstrap import *
sys.path.insert(0,str(BASE/'work/deps'));sys.path.insert(0,str(W/'raw_a3_candidate'))
import numpy as np,torch
from pooling import target_neighborhood
from dataset import read_manifest
from uni2 import load_uni2
from crop import centered_crop,image_tensor
from PIL import Image
torch.set_num_threads(4);P=R/'EXPLORATION_4';rows=read_manifest(OLD/'01_sample_definition/sample_manifest.csv');ids=[0,449];patch=np.load(P/'TOKEN_AUDIT/CACHE/patch.npy',mmap_mode='r');prefix=np.zeros((2,9,1536),np.float32);tokens=torch.from_numpy(np.concatenate([prefix,np.asarray(patch[ids])],1));points=[[rows[i]['x'],rows[i]['y']] for i in ids];reference=np.load(P/'TARGET_POOLING/REPRESENTATIONS/A3.npy')[ids]
z=target_neighborhood(tokens,points).numpy();err=float(np.abs(z-reference).max());assert np.allclose(z,reference,atol=2e-5,rtol=1e-5)
m=load_uni2(r'D:\Research\PUMA\Code\PUMA_pretrained_checkpoints\UNI2-h\uni2_h_model.bin').eval().requires_grad_(False);xs=[]
for i in ids:
 r=rows[i]
 with Image.open(r['image']) as im:xs.append(image_tensor(centered_crop(im.convert('RGB'),r['x'],r['y'],96)))
with torch.inference_mode():fresh=target_neighborhood(m.forward_features(torch.stack(xs)).float(),points).numpy()
rawerr=float(np.abs(fresh-reference).max());assert np.allclose(fresh,reference,atol=3e-4,rtol=3e-4),rawerr
dump(W/'raw_a3_candidate/RAW_FORWARD_VERIFICATION.json',{'indices':ids,'UIDs':[rows[i]['uid'] for i in ids],'cached_pooling_max_error':err,'raw_image_full_encoder_max_error':rawerr,'pass':True,'scope':'two raw RGB nuclei and exact executed cached pooling, full frozen UNI2 forward; not new training'})
print('RAW_A3_VERIFIED',err,rawerr)
