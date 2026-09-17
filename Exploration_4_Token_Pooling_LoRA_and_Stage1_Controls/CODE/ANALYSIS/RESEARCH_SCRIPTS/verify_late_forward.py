from bootstrap import *
import sys,subprocess,os
# Add the new cohort adapter to the already-used read-only verifier in memory.
src=(W/'check_package_forward.py').read_text();src=src.replace(" if package.name=='C_ALIGNMENT':"," if package.name.startswith('F_'):\n  ix=np.array(json.loads((engine.P/'BIOMASK_GUIDED/REPRESENTATIONS/contract.json').read_text())['original_indices']);bio=np.load(engine.OLD/'02_cache/biology/tierA.npy')[ix];mean=np.array(ck['normalizer']['mean'],dtype='float32');std=np.array(ck['normalizer']['std'],dtype='float32');h=torch.from_numpy(np.load(engine.P/'BIOMASK_GUIDED/REPRESENTATIONS'/(cfg['family']+'.npy')));x=torch.cat([h,torch.from_numpy(((bio-mean)/std).astype('float32'))],1)\n elif package.name=='C_ALIGNMENT':")
target=W/'check_late_forward_generated.py';target.write_text(src)
for d in sorted((R/'EXPLORATION_4/ARCHITECTURES').iterdir()):
 if d.name not in ['F_A3','F_GAUSSIAN','F_MASK','F_CLS_MASK','F_SHUFFLE','F_CLS_SHUFFLE','FINAL_A3']:continue
 if (d/'PROVENANCE/checkpoint_forward_verification.json').exists():continue
 rr=subprocess.run([sys.executable,str(target),str(d)],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});print(rr.stdout,rr.stderr,flush=True);assert rr.returncode==0
