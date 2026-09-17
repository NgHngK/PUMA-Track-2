import subprocess,sys,json,os,datetime
from bootstrap import *
if __name__=='__main__':
 packages=sorted([p.parent.parent for p in R.glob('PROMPT_*/ARCHITECTURES/*/CODE/parity_test.py')]);out=[]
 for d in packages:
  prov=d/'PROVENANCE';prov.mkdir(exist_ok=True)
  if (prov/'standalone_verification.json').exists():out.append(json.loads((prov/'standalone_verification.json').read_text()));continue
  # Preserve the generated initial dependency declaration and correct it to the measured runtime.
  req=d/'requirements.txt';cp(req,prov/'requirements_initial_draft.txt');req.write_text('torch==2.11.0\nnumpy==2.2.6\nscipy\nscikit-learn\nPillow\npsutil\ntimm==1.0.20\nscikit-image\nmatplotlib\n')
  commands=[['parity_test.py'],['run_train.py','--help'],['run_eval.py','--help']];results=[]
  for cmd in commands:
   p=d/'CODE'/cmd[0];arg=str(p.resolve());arg=arg if arg.startswith('\\\\?\\') else '\\\\?\\'+arg
   rr=subprocess.run([sys.executable,arg,*cmd[1:]],capture_output=True,text=True,env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'});results.append({'entrypoint':cmd[0],'returncode':rr.returncode,'stdout':rr.stdout,'stderr':rr.stderr})
  record={'package':str(d.relative_to(R)),'tests':results,'pass':all(v['returncode']==0 for v in results),'scope':'localV17 fixtures and CLI startup, not a rerun of all archived training','timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat()};dump(prov/'standalone_verification.json',record);out.append(record);print(d.name,record['pass'],flush=True)
 dump(R/'HISTORICAL_PACKAGE_VERIFICATION.json',out)
