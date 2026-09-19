from pathlib import Path
import json,sys,subprocess
R=Path(r'C:\Users\Hngk\Documents\Codex');C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE';P=R/'PUMA_STAGE2_PROMPT6/DATA_PRESELECTION';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
sys.path.insert(0,str(C/'src'))
from puma_exploration6.config import ExperimentConfig
base=json.loads((O/'CONFIGS_RESOLVED/A_D900_base.json').read_text())
base.update(cached_features=None,uni2_weights=r'D:\Research\PUMA\Code\PUMA_pretrained_checkpoints\UNI2-h\uni2_h_model.bin')
base['optimizer'].update(batch_size=1,accumulation_steps=64,amp=False)
names=['AUG0','AUG1','AUG2'];aug=[{},dict(geometric=True),dict(geometric=True,stain_mode='fixed_hed',stain_strength=.08)]
for name,a in zip(names,aug):
 d=json.loads(json.dumps(base));d.update(experiment_id=f'p6_B_{name}',augmentation=a);cfg=ExperimentConfig.from_dict(d)
 p=O/f'CONFIGS_RESOLVED/B_{name}_base.json';p.write_text(json.dumps(cfg.to_dict(),indent=2),encoding='utf-8')
 subprocess.run([sys.executable,str(C/'scripts/make_seed_replicates.py'),'--base',str(p),'--out-dir',str(O/f'CONFIGS_RESOLVED/B_AUGMENTATION/{name}')],check=True,cwd=C)
print('Prepared nine matched raw augmentation runs; fixed10, effective batch64, UNI2 frozen.')
