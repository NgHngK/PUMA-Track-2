import sys,time,json
from pathlib import Path
sys.path[:0]=[str(Path(__file__).parent/'deps'),str(Path(__file__).parent.parent/'outputs')]
import torch
from model import load_uni2
torch.set_num_threads(4)
t=time.time();m=load_uni2(r'D:\Research\PUMA\Code\PUMA_pretrained_checkpoints\UNI2-h\uni2_h_model.bin').eval()
print('load_seconds',time.time()-t,flush=True)
times=[]
with torch.inference_mode():
    for i in range(2):
        t=time.time();y=m(torch.zeros(1,3,224,224));times.append(time.time()-t)
        print('forward',i,times[-1],tuple(y.shape),flush=True)
(Path(__file__).parent/'uni2_benchmark.json').write_text(json.dumps({'forward_seconds':times,'shape':list(y.shape)}))
