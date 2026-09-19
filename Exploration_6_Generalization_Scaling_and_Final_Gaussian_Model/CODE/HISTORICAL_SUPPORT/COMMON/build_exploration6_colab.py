from pathlib import Path
import shutil,json,hashlib,datetime
R=Path.cwd(); P=R/'PUMA_STAGE2_COLAB'; O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
P.mkdir(exist_ok=True)
def copytree(a,b):
 shutil.copytree(a,b,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__','.pytest_cache','build','.coverage'))
copytree(R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE',P/'CODE')
copytree(R/'PUMA_STAGE2_PROMPT6/DATA_PRESELECTION',P/'INPUTS/DATA_PRESELECTION')
for name in ['01_training_dataset_tif_ROIs','01_training_dataset_geojson_nuclei']:
 copytree(Path('D:/Research/PUMA/Code/TRAINING CODE/Dataset')/name,P/'INPUTS/RAW'/name)
(P/'INPUTS/WEIGHTS').mkdir(parents=True,exist_ok=True)
shutil.copy2('D:/Research/PUMA/Code/PUMA_pretrained_checkpoints/UNI2-h/uni2_h_model.bin',P/'INPUTS/WEIGHTS/uni2_h_model.bin')
for name in ['CONFIGS_RESOLVED','RESEARCH_STATE','PROVENANCE','TABLES','REPORTS','FIGURES','HPO','DATASET_X3','EXPERIMENTS','INTERRUPTED_ATTEMPTS']:
 if (O/name).exists():copytree(O/name,P/'CONTEXT'/name)
(P/'scripts').mkdir(exist_ok=True)
s= (R/'prompt6_summarize_augmentation.py').read_text()
s=s.replace("R=Path(r'C:\\Users\\Hngk\\Documents\\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'", "import sys\nO=Path(sys.argv[1])")
(P/'scripts/summarize_augmentation.py').write_text(s)
print('Copies complete',flush=True)
