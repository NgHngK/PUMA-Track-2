from bootstrap import *
import collections,re,datetime
initial=list(csv.DictReader((R/'MASTER_ARTIFACT_INDEX.csv').open(encoding='utf8')));original=[]
for row in initial:
 p=Path(row['original_path']);dest=R/row['new_archival_path'];a=sha(p) if p.exists() else 'MISSING';b=sha(dest);original.append({'original_path':str(p),'archival_path':row['new_archival_path'],'initial_sha256':row['file_hash'],'current_original_sha256':a,'copy_sha256':b,'pass':a==b==row['file_hash']})
csvout(R/'ORIGINAL_IMMUTABILITY_VERIFICATION.csv',original)
assert all(r['pass'] for r in original),'Original or preserved copy changed'
lookup={str((R/r['new_archival_path']).resolve()).lower():r for r in initial};records=[];hashes=collections.defaultdict(list)
root=Path('\\\\?\\'+str(R.resolve()))
exclude={'MASTER_ARTIFACT_INDEX_COMPLETE.csv','ARCHIVE_VERIFICATION_REPORT.json','ARCHIVE_VERIFICATION_REPORT.md','ARCHIVE_FILE_MANIFEST.json','ARCHIVE_SHA256.txt'}
for p in sorted(root.rglob('*')):
 if not p.is_file() or '.git' in p.parts or '__pycache__' in p.parts or p.name in exclude:continue
 rel=p.relative_to(root);parts=rel.parts;normal=str(R/rel).lower();old=lookup.get(normal,{});digest=sha(p);hashes[digest].append(str(rel));n=next((int(x[-1]) for x in parts if re.fullmatch('PROMPT_[1-4]',x)),None);arch=parts[parts.index('ARCHITECTURES')+1] if 'ARCHITECTURES' in parts else old.get('architecture_id','SHARED_OR_INDEX');cfg={};run=''
 if 'RUNS' in parts:
  j=parts.index('RUNS');run=parts[j+1];q=root.joinpath(*parts[:j+2])/'config.json'
  if q.exists():cfg=json.loads(q.read_text(encoding='utf8'))
 records.append({'archival_path':str(rel).replace('\\','/'),'sha256':digest,'bytes':p.stat().st_size,'original_path':old.get('original_path','see package ORIGINAL_PATH_MAP or provenance'),'prompt_number':n,'architecture_id':arch,'experiment_id':cfg.get('id',run or old.get('experiment_id','NOT APPLICABLE')),'seed':cfg.get('seed',old.get('seed','NOT RECORDED')),'fold':cfg.get('fold',old.get('fold','NOT RECORDED')),'epoch':re.search(r'epoch_(\d+)',p.name).group(1) if re.search(r'epoch_(\d+)',p.name) else old.get('epoch','see history/config'),'population':cfg.get('population',old.get('data_population','see manifest/provenance')),'coordinate_source':cfg.get('coordinate_source',cfg.get('train_coordinates',old.get('coordinate_source','see manifest'))),'FOV':cfg.get('fov',96 if n in [3,4] and run else old.get('FOV','see config')),'loss':cfg.get('loss','ordinary CE; see config' if n in [3,4] and run else old.get('loss','see config')),'sampler':cfg.get('sampler',old.get('sampler','see config')),'checkpoint_role':cfg.get('selection',cfg.get('role',old.get('checkpoint_role','see fixed endpoint or original selection'))),'result_type':p.suffix,'status':old.get('current_status','VERIFIED ARCHIVAL FILE; attribution from package/provenance')})
csvout(R/'MASTER_ARTIFACT_INDEX_COMPLETE.csv',records);dump(R/'ARCHIVE_FILE_MANIFEST.json',{r['archival_path']:{'sha256':r['sha256'],'bytes':r['bytes']} for r in records})
duplicates=[{'sha256':h,'copies':len(ps),'paths':ps} for h,ps in hashes.items() if len(ps)>1];dump(R/'DUPLICATE_FILE_LINEAGE.json',duplicates)
packages=[]
for n in range(1,5):
 for d in (R/f'PROMPT_{n}/ARCHITECTURES').iterdir():
  if not d.is_dir():continue
  required=['ARCHITECTURE.md','FLOWCHART.md','README.md','CODE/run_train.py','CODE/run_eval.py','config.json','requirements.txt','RESULT_INDEX.csv','ORIGINAL_PATH_MAP.csv'];missing=[x for x in required if not (d/x).exists()];packages.append({'package':str(d.relative_to(R)),'missing_required':missing,'status':'UNRESOLVED HISTORICAL EXECUTABLE' if 'UNRESOLVED' in d.name or 'EXCLUDED' in d.name else ('COMPLETE_LAYOUT' if not missing else 'MISSING')})
verification=json.loads((R/'HISTORICAL_PACKAGE_VERIFICATION.json').read_text());report={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'original_files_checked':len(original),'originals_unchanged':all(r['pass'] for r in original),'archival_files_indexed':len(records),'total_bytes':sum(r['bytes'] for r in records),'duplicate_hash_groups':len(duplicates),'packages':packages,'standalone_tests':len(verification),'standalone_all_pass':all(r['pass'] for r in verification),'unresolved_original_attributions':sum('UNRESOLVED' in r['status'] for r in records),'final_architecture':'Exploration3 A5 retained after Exploration4 A3 failed recall guard','index_scope':'excludes .git, bytecode and self-referential audit files; final ZIP verified separately'}
dump(R/'ARCHIVE_VERIFICATION_REPORT.json',report);print(json.dumps({k:v for k,v in report.items() if k!='packages'},indent=2));print('Missing package files',[r for r in packages if r['missing_required']])
