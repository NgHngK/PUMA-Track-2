import sys,json,copy,csv,tempfile
from pathlib import Path
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT.parent/'outputs'))
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader,TensorDataset
from PIL import Image
from dataset import centered_crop,CLASSES,read_manifest
from model import QVLoRA,load_uni2
from loss import LogitAdjustedCE
from train_eval import metrics,train_epoch

torch.set_num_threads(1);torch.manual_seed(17);checks={}
image=Image.new('RGB',(20,20),'white');image.putpixel((0,0),(0,0,0))
c=np.asarray(centered_crop(image,0,0,8));assert (c[4,4]==0).all() and (c[0,0]==255).all();checks['edge_center_preserved']=True
prior=torch.tensor([.6971,.1363,.006,.0684,.0078,.0024,.0285,.0253,.0175,.0107]);prior/=prior.sum()
z=torch.randn(11,10,requires_grad=True);y=torch.arange(11)%10;la=LogitAdjustedCE(prior,1)
l=la(z,y);ref=(-z[torch.arange(11),y]-prior.log()[y]+torch.logsumexp(z+prior.log(),-1)).mean()
assert torch.allclose(l,ref);g=torch.autograd.grad(l,z)[0];expected=((z+prior.log()).softmax(-1)-nn.functional.one_hot(y,10))/len(y)
assert torch.allclose(g,expected,atol=1e-7);checks['balanced_softmax_and_gradient']=True
base=nn.Linear(12,36);qv=QVLoRA(base,2,4);x=torch.randn(3,5,12)
assert torch.equal(base(x),qv(x));qv(x).square().mean().backward()
assert qv.bq.grad.norm()>0 and qv.aq.grad.norm()==0 and base.weight.grad is None
opt=torch.optim.SGD([p for p in qv.parameters() if p.requires_grad],lr=.1);opt.step();opt.zero_grad();qv(x).square().mean().backward();assert qv.aq.grad.norm()>0
checks['lora_zero_B_initialization_and_delayed_A_gradient']=True
xs=torch.randn(11,4);ys=torch.arange(11)%10;m=nn.Linear(4,10);m2=copy.deepcopy(m)
o=torch.optim.SGD(m.parameters(),lr=.001);o2=torch.optim.SGD(m2.parameters(),lr=.001)
criterion=LogitAdjustedCE(torch.ones(10),0);ds=TensorDataset(xs,ys,torch.arange(11))
train_epoch(m,DataLoader(ds,batch_size=4),o,criterion,torch.device('cpu'),accum=3)
train_epoch(m2,DataLoader(ds,batch_size=11),o2,criterion,torch.device('cpu'))
assert all(torch.allclose(a,b,atol=1e-7) for a,b in zip(m.parameters(),m2.parameters()));checks['partial_accumulation_equivalence']=True
me=metrics(np.array([0,0]),torch.tensor([[20.]+[0.]*9]*2));assert abs(me['macro_f1']-.1)<1e-6 and len(me['missing_classes'])==9;checks['fixed10_missing_classes']=True
try:load_uni2(r'D:\Research\PUMA\Code\TRAINING CODE\pytorch_model.bin')
except ValueError:checks['wrong_checkpoint_rejected']=True
assert checks.get('wrong_checkpoint_rejected')
with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
    folder=Path(tmp);image.save(folder/'tile.png')
    fields=['uid','roi','group','image','x','y','label','class_name','split','coordinate_source']
    rows=[dict(zip(fields,['a','r1','g1','tile.png',1,1,0,'tumor','train','gt'])),
          dict(zip(fields,['b','r2','g2',str(folder/'tile.png'),1,1,0,'tumor','val','gt']))]
    with (folder/'manifest.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    try:read_manifest(folder/'manifest.csv')
    except ValueError as e:assert 'image leakage' in str(e);checks['absolute_relative_image_leakage_rejected']=True
assert checks.get('absolute_relative_image_leakage_rejected')
report={'checks':checks,'batch64_neutrophil_absence_ratio_text':float((1-prior[5])**64),'batch64_neutrophil_absence_local':float((1-366/97193)**64)}
(ROOT/'integrity_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
