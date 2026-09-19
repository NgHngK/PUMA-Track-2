from pathlib import Path
import csv, json, hashlib, sys, platform, subprocess, os
ROOT=Path(r'C:\Users\Hngk\Documents\Codex')
CODE=ROOT/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE'
MASTER=ROOT/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4'
OUT=MASTER/'EXPLORATION_6'
sys.path.insert(0,str(CODE/'src'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj): (OUT/p).write_text(json.dumps(obj,indent=2),encoding='utf-8')
records=[]
for line in (CODE/'SHA256SUMS.txt').read_text().splitlines():
    h,p=line.split(None,1); f=CODE/p
    records.append(dict(path=p,expected=h,actual=sha(f) if f.exists() else None))
save('PROVENANCE/SOURCE_HASH_VERIFICATION.json',records)
backup=MASTER/'MASTER_REPORT_PROMPTS_1_TO_4_PRE_PROMPT6_BACKUP.md'
original=MASTER/'MASTER_REPORT_PROMPTS_1_TO_4.md'
if not backup.exists(): backup.write_bytes(original.read_bytes())
save('PROVENANCE/MASTER_REPORT_APPEND_INTEGRITY.json',dict(status='BACKED_UP_NOT_APPENDED',original_sha256=sha(original),backup_sha256=sha(backup),original_byte_length=original.stat().st_size,prefix_byte_identity=original.read_bytes()==backup.read_bytes(),appended_byte_length=0,prompt5_content_identity=None,prompt6_content_identity=None))
from puma_exploration6.config import ExperimentConfig
configs=[]
for p in sorted((CODE/'configs').glob('*.json')):
    try: ExperimentConfig.load(p); configs.append(dict(path=str(p),status='PASS'))
    except Exception as e: configs.append(dict(path=str(p),status='FAIL',error=str(e)))
save('PROVENANCE/CONFIG_VALIDATION.json',configs)
compile_status=subprocess.run([sys.executable,'-m','compileall','-q','src','scripts','tests','vendor'],cwd=CODE,capture_output=True,text=True)
help_results=[]
for p in sorted((CODE/'scripts').glob('*.py')):
    if p.name.startswith('_'): continue
    r=subprocess.run([sys.executable,str(p),'--help'],cwd=CODE,capture_output=True,text=True)
    (OUT/'PROVENANCE'/f'{p.stem}_help.txt').write_text(r.stdout+r.stderr,encoding='utf-8')
    help_results.append(dict(script=p.name,returncode=r.returncode))
import torch
save('PROVENANCE/CODEBASE_AUDIT.json',dict(code_root=str(CODE),source_records=len(records),source_matches_shipped_sha=all(r['actual']==r['expected'] for r in records),sha_manifest_sha256=sha(CODE/'SHA256SUMS.txt'),configs=configs,compile_returncode=compile_status.returncode,cli_help=help_results,python=sys.version,executable=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,cuda_available=torch.cuda.is_available(),platform=platform.platform(),tests_status='RUNNING_SEPARATELY',deviations=[]))
hist=list(csv.DictReader((ROOT/'PUMA_STAGE2_PROMPT5/SPLIT_MANIFEST_AUDIT.csv').open(newline='',encoding='utf-8-sig')))
seen={r['roi'] for r in hist}
dataset=Path(r'D:\Research\PUMA\Code\TRAINING CODE\Dataset')
counts={}; props=set()
for p in (dataset/'01_training_dataset_geojson_nuclei').glob('*.geojson'):
    roi=p.stem.removesuffix('_nuclei'); cc={}
    for f in json.loads(p.read_text())['features']:
        props.update(f.get('properties',{}))
        c=f['properties']['classification']['name'].removeprefix('nuclei_'); cc[c]=cc.get(c,0)+1
    counts[roi]=cc
classes=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
fresh=set(counts)-seen
support=[dict(class_name=c,remaining_nuclei=sum(counts[r].get(c,0) for r in fresh),remaining_rois=sum(counts[r].get(c,0)>0 for r in fresh),locked_required=q) for c,q in zip(classes,[59,22]+[18]*8)]
with (OUT/'DATASET_X3/FRESH_GROUP_CLASS_SUPPORT.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(support[0]));w.writeheader();w.writerows(support)
save('DATASET_X3/HISTORICAL_OVERLAP_PREFLIGHT.json',dict(total_rois=len(counts),historical_screen_rois=len(seen),remaining_rois=len(fresh),historical_manifest=str(ROOT/'PUMA_STAGE2_PROMPT5/SPLIT_MANIFEST_AUDIT.csv'),historical_manifest_sha256=sha(ROOT/'PUMA_STAGE2_PROMPT5/SPLIT_MANIFEST_AUDIT.csv'),remaining_roi_ids=sorted(fresh),support=support,geojson_property_keys=sorted(props),note='Excludes only the historical 450-nucleus cohort groups; full historical exposure may exclude more. This is an optimistic upper bound, not certification of freshness.'))
print(json.dumps(dict(config_pass=sum(c['status']=='PASS' for c in configs),cli_pass=sum(r['returncode']==0 for r in help_results),compile_returncode=compile_status.returncode,fresh_roi_support=support)))
