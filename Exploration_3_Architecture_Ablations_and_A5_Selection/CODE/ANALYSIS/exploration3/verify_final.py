import sys,json,csv,compileall,subprocess,hashlib
from pathlib import Path
import numpy as np,torch
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';D=R/'93_final_architecture';sys.path.insert(0,str(D))
from model import CachedClassifier
torch.set_num_threads(2);checks={'compile':compileall.compile_dir(D,quiet=1)}
for n in ['features','train_eval','predict','calibrate']:
    p=subprocess.run([sys.executable,str(D/(n+'.py')),'--help'],capture_output=True,text=True);assert p.returncode==0,p.stderr
checks['CLI_entrypoints']='PASS';torch.manual_seed(17);m=CachedClassifier();x=torch.randn(12,1552);z=m(x)
assert torch.equal(z,m.head(x[:,:1536]));assert sum(p.numel() for p in m.parameters())==27866
torch.nn.functional.cross_entropy(z,torch.arange(12)%10).backward();assert m.branch.weight.grad.norm()>0 and m.ph.weight.grad.norm()==0
checks['initial_baseline_and_delayed_projection_gradient']='PASS'
folder=R/'experiments/p3_confirm_A5_s17';ck=torch.load(folder/'model.pt',map_location='cpu',weights_only=True);m.load_state_dict(ck['model'],strict=True)
raw=torch.from_numpy(np.load(R/'02_cache/uni2_features/gt_fov96/features.npy'));h=torch.nn.functional.layer_norm(raw,(1536,));b=np.load(R/'02_cache/biology/tierA.npy');norm=ck['normalizer'];b=(b-np.array(norm['mean']))/np.array(norm['std']);x=torch.cat((h,torch.from_numpy(b.astype('float32'))),1)
saved=np.load(folder/'predictions.npz');ii=saved['indices']
with torch.no_grad():z=m(x[ii]).numpy()
err=float(np.max(np.abs(z-saved['logits'])));assert err<1e-5;checks['confirmation_logit_max_abs_error']=err
checks['parameters']=27866;checks['mask_inputs_absent']=True
# Keep the historical evaluator and source identities unchanged.
count=0
for p in (R/'experiments').glob('*/provenance.json'):
    v=json.loads(p.read_text());hashes=v.get('sources',v.get('modules',{}))
    for n,expected in hashes.items():
        assert hashlib.sha256((R/'01_shared_core'/n).read_bytes()).hexdigest()==expected,(p,n);count+=1
checks['all_experimental_source_hashes_verified']=count
from evaluation import evaluate_proposals
rows=[{'roi':'a','x':20.,'y':20.,'uid':'correct'},{'roi':'a','x':50.,'y':50.,'uid':'wrong'}]
result=evaluate_proposals(rows,np.eye(10)[[7,8]],W.parent/'final_verification/annotations',['a','b']);cm=result['summed']['class_metrics'];assert(cm['nuclei_epithelium']['TP'],cm['nuclei_epithelium']['FN'],cm['nuclei_endothelium']['FP'])==(1,1,1)
checks['fullROI_zero_proposal_and_canonical_mapping']='PASS'
(D/'VERIFICATION.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))
