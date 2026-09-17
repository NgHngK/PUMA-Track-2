from bootstrap import *
import subprocess,sys,os
for d in sorted((R/'EXPLORATION_4/ARCHITECTURES').iterdir()):
 if not (d/'CODE/engine.py').exists() or (d/'PROVENANCE/checkpoint_forward_verification.json').exists():continue
 rr=subprocess.run([sys.executable,str(W/'check_package_forward.py'),str(d)],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1'});print(rr.stdout,rr.stderr,flush=True)
 if rr.returncode:raise RuntimeError(d.name)
