import csv,json,hashlib,shutil,datetime,sys
from pathlib import Path
W=Path(__file__).resolve().parent;BASE=W.parents[1];OLD=BASE/'outputs/STAGE2_RESEARCH_20260907';R=BASE/'outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4'
def sha(p):
 p=p if str(p).startswith('\\\\?\\') else Path('\\\\?\\'+str(p.resolve()))
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,o):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(o,indent=2),encoding='utf8')
def cp(s,d):
 s=Path('\\\\?\\'+str(s.resolve()));d=Path('\\\\?\\'+str(d.resolve()))
 d.parent.mkdir(parents=True,exist_ok=True)
 if d.exists():assert sha(s)==sha(d)
 else:shutil.copy2(s,d)
def csvout(p,rs):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
if __name__=='__main__':
 for n in range(1,5):
  for d in ['EXPLORATION_TEXT','ARCHITECTURES','SHARED_INPUT_SNAPSHOTS','EVALUATION','FIGURES','TABLES','PROVENANCE']:(R/f'PROMPT_{n}'/d).mkdir(parents=True,exist_ok=True)
 for d in ['TOKEN_AUDIT','TARGET_POOLING','MULTISCALE','STAGE1_PROPOSALS','ROI_CLASS_TRAINING','LORA','BIOMASK_GUIDED']:(R/'EXPLORATION_4'/d).mkdir(exist_ok=True)
 cp(Path('C:/Users/Hngk/Downloads/exploration4.md'),R/'EXPLORATION_4/EXPLORATION_TEXT/original_exploration.md')
 cp(OLD/'00_reference/RESEARCH_STATE.json',R/'EXPLORATION_3/PROVENANCE/RESEARCH_STATE_FROZEN.json')
 records=[];lineage=[]
 for p in sorted(BASE.rglob('*')):
  if not p.is_file() or R in p.parents or W in p.parents:continue
  rel=p.relative_to(BASE);parts=rel.parts
  if any(x in parts for x in ['.git','__pycache__','deps','.agents','.codex']):continue
  if parts[0] not in ['outputs','work']:continue
  if p.suffix in ['.b64','.whl']:continue
  n=1;arch='UNRESOLVED';cfg={};eid='UNKNOWN';status='UNRESOLVED — DO NOT ATTRIBUTE'
  if OLD in p.parents:
   n=2
   if 'experiments' in parts:
    j=parts.index('experiments');ed=OLD/'experiments'/parts[j+1] if len(parts)>j+2 else None
    if ed and (ed/'config.json').exists():
     cfg=json.loads((ed/'config.json').read_text());eid=cfg['id'];n=3 if 'architecture' in cfg else 2
     arch=cfg.get('architecture',('BIOLOGY_ONLY' if cfg.get('biology_only') else 'FROZEN_UNI2')+'_'+cfg.get('biology','none'))
     status='historical executed; see exact config'
   elif any(s in str(rel).lower() for s in ['exploration3','biomask','tierb','architecture_','p3_']):n=3
   elif '93_final_architecture' in parts:n=3;arch='A5';status='Exploration3 selected standalone'
  elif len(parts)>1 and parts[1]=='exploration3':n=3
  elif p.parent.name in ['cnn_results','uni2_results','lora_results','lora_results_v2']:
   arch={'cnn_results':'CNN','uni2_results':'UNI2_PROBE','lora_results':'LORA_EXCLUDED','lora_results_v2':'LORA_CORRECTED'}[p.parent.name];status='historical family from execution directory'
  elif len(parts)>1 and (parts[1].startswith('continue_') or parts[1].startswith('CONTINUATION') or parts[1] in ['final_code','final_verification','standalone_smoke']):n=2
  # The index retains exact source relative paths inside family evidence, preventing collisions.
  shared=any(t in str(rel).lower() for t in ['02_cache','pixels.npz','features.npy','sample.csv','manifest.csv'])
  bucket='SHARED_INPUT_SNAPSHOTS' if shared else (f'ARCHITECTURES/{arch}/RESULTS/ORIGINAL' if arch!='UNRESOLVED' else 'UNRESOLVED_HISTORICAL_ARTIFACTS')
  dest=R/f'PROMPT_{n}'/bucket/rel
  h=sha(p);cp(p,dest)
  records.append(dict(original_path=str(p),file_name=p.name,file_hash=h,prompt_number=n,architecture_id=arch,experiment_id=eid,seed=cfg.get('seed','UNKNOWN'),fold=cfg.get('fold','UNKNOWN'),epoch=cfg.get('epochs','UNKNOWN'),data_population='see source manifest',coordinate_source='see provenance',FOV=cfg.get('fov',96 if n==3 and eid!='UNKNOWN' else 'UNKNOWN'),loss=cfg.get('tau','UNKNOWN'),sampler=cfg.get('sampler','UNKNOWN'),checkpoint_role='see original filename/config',result_type=p.suffix,current_status=status,new_archival_path=str(dest.relative_to(R)),bytes=p.stat().st_size,copy_sha256=sha(dest),shared_cache=shared))
  if eid!='UNKNOWN' and not any(x['id']==eid for x in lineage):lineage.append({'id':eid,'prompt':n,'architecture':arch,'config':cfg,'source':str(p.parent)})
 csvout(R/'MASTER_ARTIFACT_INDEX.csv',records);dump(R/'MASTER_PROVENANCE.json',{'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'originals_immutable':True,'inventory_scope':'workspace outputs and work, excluding interpreter deps/git/pycache/transport base64; external inputs referenced by original provenance','files':len(records),'bytes':sum(r['bytes'] for r in records),'experiments':lineage})
 (R/'MASTER_CODE_LINEAGE.md').write_text('# Historical code lineage\n\nRecovered before Exploration4 experiments. Original paths and complete configs are preserved in MASTER_PROVENANCE.json and MASTER_ARTIFACT_INDEX.csv. UNKNOWN means attribution not proven.\n\n'+ '\n'.join(f"- Exploration{x['prompt']} / {x['architecture']} / {x['id']}: `{json.dumps(x['config'])}`" for x in lineage),encoding='utf8')
 dump(W/'research-state.json',{'status':'active','phase':'history copied and indexed; protocol and token audit next','root':str(R),'old_root':str(OLD),'completed_runs':[]})
 (W/'findings.md').write_text('# Exploration4 findings\n\nExploration3 A5 remains reference; previous stop on head search is superseded by explicit Exploration4 representation study. Historical sources are immutable.\n',encoding='utf8')
 print('Copied',len(records),'files;',len(lineage),'configured experiments;',sum(r['bytes'] for r in records),'bytes',flush=True)
