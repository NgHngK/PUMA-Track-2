from pathlib import Path
import json,sys,subprocess,datetime,hashlib
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
sys.path.insert(0,str(C/'src'))
from puma_exploration6.config import ExperimentConfig
amend=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),id='SCHEDULING_AMENDMENT_01',evidence='Measured 1350 deterministic raw forwards took about29 minutes on CPU; no supported CUDA hardware identified. Nine Group-A controls complete; Group-B raw parity active.',change='Run fixed-recipe cached CLS Group-C controls during raw-image computation. No final sampling recipe will be chosen until Group-B is resolved.',scientific_status='Conditional frozen-CLS no-augmentation controls, not an augmentation-integrated winner',bounded_arms=['natural CE','tempered alpha0.5 CE','cRT reset W/R frozen Ph/Pb 5 extra epochs','matched cRT no-reset W/R frozen Ph/Pb 5 extra epochs'],control='D900 historical inverse CE fixed10 already executed; reuse existing runs',seeds=[17,29,43],decision='Do not open confirmation/natural; if augmentation is retained, verify relevant sampling candidate under retained augmentation before any promotion.',other_stages='No encoder adaptation, multiscale or old LoRA reruns authorized by this amendment')
(O/'RESEARCH_STATE/SCHEDULING_AMENDMENT_01.json').write_text(json.dumps(amend,indent=2),encoding='utf-8')
base=json.loads((O/'CONFIGS_RESOLVED/A_D900_base.json').read_text())
for name in ['natural','tempered','crt_reset','crt_noreset']:
 d=json.loads(json.dumps(base));d['experiment_id']=f'p6_C_cached_{name}'
 if name in ['natural','tempered']:d['sampler'].update(mode=name,alpha=0.0 if name=='natural' else .5)
 else:d['decoupled'].update(enabled=True,reset_classifier=name=='crt_reset',retrain_epochs=5,retrain_lr=.001)
 cfg=ExperimentConfig.from_dict(d);p=O/f'CONFIGS_RESOLVED/C_{name}_base.json';p.write_text(json.dumps(cfg.to_dict(),indent=2),encoding='utf-8')
 dest=O/f'CONFIGS_RESOLVED/C_CACHED/{name}'
 subprocess.run([sys.executable,str(C/'scripts/make_seed_replicates.py'),'--base',str(p),'--out-dir',str(dest)],cwd=C,check=True)
 with (O/f'PROVENANCE/C_{name}_batch.log').open('w',encoding='utf-8') as log:
  subprocess.run([sys.executable,str(C/'scripts/run_batch.py'),'--config-dir',str(dest),'--stop-on-error'],cwd=C,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Twelve conditional cached sampling/cRT controls complete. Holdouts unopened.')
