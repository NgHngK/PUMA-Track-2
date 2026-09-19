from pathlib import Path
import csv,json,hashlib,shutil
R=Path(r'C:\Users\Hngk\Documents\Codex');H=R/'PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_1/UNRESOLVED_HISTORICAL_ARTIFACTS/work/ingestion';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6/TABLES/HISTORICAL_IMPORTED_UNRESOLVED';O.mkdir(exist_ok=True)
index=[];allrows={}
for name in ['drive_history.csv','history_compact.csv']:
 p=H/name;q=O/name
 if q.exists():assert q.read_bytes()==p.read_bytes()
 else:shutil.copy2(p,q)
 with p.open(newline='',encoding='utf-8-sig') as f:
  rd=csv.DictReader(f);rows=list(rd);fields=rd.fieldnames
 allrows[name]=rows
 index.append(dict(source=str(p),copy=str(q),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),epochs=len(rows),fields=len(fields),status='UNRESOLVED — DO NOT ATTRIBUTE to A5 or new Exploration6 data',note='Recorded class fields are mapped by class name, never positional epithelium/endothelium order. Preserve original evaluator labels; parity and independent provenance are not inferred.'))
 with (O/(p.stem+'_ALL_RECORDED_FIELDS.csv')).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=['source','epoch','field','recorded_value','status']);w.writeheader()
  for row in rows:
   for field,value in row.items():w.writerow(dict(source=name,epoch=row['epoch'],field=field,recorded_value=value,status='UNRESOLVED HISTORICAL IMPORT'))
# Compact CSV is a projection; identify overlap numerically without erasing either source.
full={r['epoch']:r for r in allrows['drive_history.csv']};matches=0;differences=[]
for r in allrows['history_compact.csv']:
 for k,v in r.items():
  a=full.get(r['epoch'],{}).get(k)
  try:equal=abs(float(a)-float(v))<=1e-12
  except (ValueError,TypeError):equal=a==v
  if equal:matches+=1
  else:differences.append(dict(epoch=r['epoch'],field=k,compact=v,full=a))
(O/'INGESTION_SUMMARY.json').write_text(json.dumps(dict(sources=index,projection_equal_fields=matches,projection_differences=differences,new_training_runs=0),indent=2),encoding='utf-8')
p=O.parent/'HISTORICAL_PROMPT1/INGESTION_SUMMARY.json';d=json.loads(p.read_text());d['exploration1']='14 canonical original JSON epoch histories; excluded and unresolved imported histories ingested separately';p.write_text(json.dumps(d,indent=2),encoding='utf-8')
print(json.dumps(dict(sources=[dict(epochs=x['epochs'],fields=x['fields']) for x in index],projection_differences=len(differences))))
