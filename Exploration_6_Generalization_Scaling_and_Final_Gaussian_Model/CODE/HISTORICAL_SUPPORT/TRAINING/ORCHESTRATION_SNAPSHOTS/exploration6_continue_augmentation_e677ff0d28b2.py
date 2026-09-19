"""Finite runner for the eight remaining preregistered Group-B configs.

Waits for the already active first run; never restarts a partial run.
No model selection or holdout evaluation is performed here.
"""
from pathlib import Path
import json,os,sys,subprocess,time,hashlib,datetime,psutil
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
status=O/'RESEARCH_STATE/AUGMENTATION_PIPELINE_STATUS.json'
def save(**kw):status.write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2),encoding='utf-8')
def verify(root,cfg):
 d=json.loads((root/'summary.json').read_text());assert json.loads((root/'config.json').read_text())==cfg
 assert d['epochs_run']==10 and d['selected_epoch']==10 and d['selection_mode']=='fixed_final'
 assert (root/d['final_model_checkpoint']).is_file()
first='p6_B_AUG0_s17';root=O/'EXPERIMENTS'/first
try:
 if not (root/'summary.json').exists():
  active=psutil.Process(17772);cmd=active.cmdline();assert any(first in s for s in cmd),cmd
  created=active.create_time();save(status='WAITING_FOR_EXISTING_RUN',run_id=first,pid=active.pid,process_create_time=created)
  while not (root/'summary.json').exists():
   if not active.is_running() or active.create_time()!=created:raise RuntimeError('Existing AUG0 process exited without a completed summary; partial run preserved')
   time.sleep(15)
 for arm in ['AUG0','AUG1','AUG2']:
  for seed in [17,29,43]:
   run=f'p6_B_{arm}_s{seed}';cp=O/f'CONFIGS_RESOLVED/B_AUGMENTATION/{arm}/{run}.json';cfg=json.loads(cp.read_text());root=O/'EXPERIMENTS'/run
   if (root/'summary.json').exists():verify(root,cfg);continue
   if root.exists():raise RuntimeError(f'Partial run exists; refusing duplicate: {root}')
   save(status='RUNNING',run_id=run,config_sha256=hashlib.sha256(cp.read_bytes()).hexdigest())
   with (O/f'PROVENANCE/{run}.log').open('x',encoding='utf-8') as log:
    subprocess.run([sys.executable,str(C/'scripts/run_experiment.py'),'--config',str(cp)],cwd=C,stdout=log,stderr=subprocess.STDOUT,check=True)
   verify(root,cfg)
 save(status='COMPLETE',runs=9,holdouts_opened=False,next_step='Aggregate three-seed augmentation results, parity and predeclared guards; then resolve conditional C controls')
 print('All nine Group-B raw augmentation runs complete; holdouts unopened.',flush=True)
except BaseException as e:
 save(status='FAILED',error=repr(e),policy='Preserve partial artifacts; no automatic retry');raise
