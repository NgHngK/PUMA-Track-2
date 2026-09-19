from pathlib import Path
import json,hashlib,shutil,ast,datetime
R=Path.cwd();P=R/'PUMA_STAGE2_COLAB';base=R/'2026-09-07/use-the-autoresearch-skill-x20-https';out=base/'outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4';s=out/'EXPLORATION_6/RESEARCH_STATE'
(P/'CONTEXT/MASTER_REPORT').mkdir(exist_ok=True)
shutil.copy2(out/'MASTER_REPORT_PROMPTS_1_TO_4.md',P/'CONTEXT/MASTER_REPORT/MASTER_REPORT_PROMPTS_1_TO_4_UNCHANGED.md')
shutil.copy2(R/'PUMA_STAGE2_PROMPT5/EXPLORATION5_ROOT_CAUSE_REPORT.md',P/'CONTEXT/MASTER_REPORT/EXPLORATION5_ROOT_CAUSE_REPORT.md')
for f in (P/'scripts').glob('*.py'):ast.parse(f.read_text(encoding='utf-8-sig'))
nb=json.loads((P/'PUMA_STAGE2_MAIN.ipynb').read_text())
for cell in nb['cells']:
 if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
# Verify each scientific queue template and all source files against original copies.
original=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE'
checked=0
for f in (P/'CODE').rglob('*'):
 if f.is_file():assert f.read_bytes()==(original/f.relative_to(P/'CODE')).read_bytes();checked+=1
queue=[]
for arm in ['AUG0','AUG1','AUG2']:
 for seed in [17,29,43]:
  if arm=='AUG0' and seed in [17,29]:continue
  cp=P/f'CONTEXT/CONFIGS_RESOLVED/B_AUGMENTATION/{arm}/p6_B_{arm}_s{seed}.json';d=json.loads(cp.read_text());assert d['optimizer']['batch_size']==1 and d['optimizer']['accumulation_steps']==64 and d['optimizer']['max_epochs']==10 and not d['optimizer']['amp'];queue.append(d['experiment_id'])
assert len(queue)==7
assert not (P/'CONTEXT/EXPERIMENTS/p6_B_AUG0_s43').exists()
assert (P/'CONTEXT/INTERRUPTED_ATTEMPTS/p6_B_AUG0_s43_attempt1/history.jsonl').is_file()
record=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='COLAB_PACKAGE_READY_NO_LOCAL_TRAINING',automation_enabled=False,queue=queue,holdouts_opened=False,validation=dict(source_files_byte_verified=checked,notebook_python_syntax='PASS',queue_contract='PASS',gpu_execution='NOT RUN — no local CUDA GPU'),recovery='User authorized continuation; original interrupted seed43 attempt archived and hash verified. Fresh unchanged seed43 config will execute on CUDA only when notebook is run.',scope='Finite B execution then required review; D–J and final master report remain pending.')
(P/'PACKAGE_VALIDATION.json').write_text(json.dumps(record,indent=2))
for q in [s/'WORKING_STATE.json',base/'work/exploration6/research-state.json']:
 d=json.loads(q.read_text());d.update(status=record['status'],current_stage='Awaiting user Colab GPU execution of remaining B queue',active_runs=[],next_required_input='Run uploaded PUMA_STAGE2_MAIN.ipynb; return COLAB_RESULTS after completion or interruption.',continuity='Manual user continuation only; no automatic wakeups. No local training active.',colab_package=str(P));q.write_text(json.dumps(d,indent=2))
for q in [s/'FINDINGS_CURRENT.md',base/'work/exploration6/findings.md']:
 with q.open('a',encoding='utf-8') as f:f.write('\n\nColab handoff: user requested GPU migration. No local trainer was launched. Seven unchanged scientific configs will run on CUDA through PUMA_STAGE2_COLAB/PUMA_STAGE2_MAIN.ipynb. Original interrupted seed43 attempt archived with verified hashes. Both holdouts remain unopened. No automatic wakeups.\n')
(P/'CONTEXT/COLAB_HANDOFF.json').write_text(json.dumps(record,indent=2))
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as x:
  for b in iter(lambda:x.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
entries=[dict(path=f.relative_to(P).as_posix(),bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(P.rglob('*')) if f.is_file() and f.name!='PACKAGE_SHA256.json']
(P/'PACKAGE_SHA256.json').write_text(json.dumps(entries,indent=2))
print(json.dumps(dict(files=len(entries),bytes=sum(e['bytes'] for e in entries),source_verified=checked,queue=queue)))
