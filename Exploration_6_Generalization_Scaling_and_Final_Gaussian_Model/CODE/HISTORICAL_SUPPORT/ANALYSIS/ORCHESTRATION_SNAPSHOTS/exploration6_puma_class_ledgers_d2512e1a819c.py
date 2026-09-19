from pathlib import Path
import json,csv,hashlib,sys
R=Path(r'C:\Users\Hngk\Documents\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
CL=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis'];M='NOT RECORDED IN ORIGINAL RUN'
groups=['HISTORICAL_PROMPT1','HISTORICAL_PROMPT1_EXCLUDED','HISTORICAL'] if '--historical' in sys.argv else ['CURRENT']
for group in groups:
 folder=O/'TABLES'/group if group!='CURRENT' else O/'TABLES';sources=[]
 if group=='CURRENT':
  for p in sorted((O/'EXPERIMENTS').glob('*/history.jsonl')):sources.append(dict(source=str(p),run_id=p.parent.name,prompt=6))
 else:sources=json.loads((folder/'HISTORY_SOURCES.json').read_text())
 fields=['prompt','run_id','phase','epoch','split','evaluation_contract_id','class_name','roi_precision','roi_recall','roi_f1','pooled_precision','pooled_recall','pooled_f1','pooled_TP','pooled_FP','pooled_FN','source_sha256']
 n=0
 with (folder/'EPOCH_PUMA_PER_CLASS_METRICS.csv').open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
  for s in sources:
   p=Path(s['source']);text=p.read_text(encoding='utf-8-sig');h=json.loads(text) if p.suffix=='.json' else [json.loads(x) for x in text.splitlines() if x.strip()];digest=hashlib.sha256(p.read_bytes()).hexdigest();phases=[('joint',h,digest)]
   sp=p.parent/'summary.json'
   if group=='CURRENT' and sp.exists():
    d=json.loads(sp.read_text());dec=(d.get('decoupled') or {}).get('epochs',[])
    if dec:phases.append(('decoupled',dec,hashlib.sha256(sp.read_bytes()).hexdigest()))
   for phase,hh,digest in phases:
    for row in hh:
     for split in ['train','val','dev']:
      if not isinstance(row.get(split),dict):continue
      pm=row[split].get('puma',{});a=pm.get('fixed10',{});b=pm.get('summed',{})
      for name in CL:
       key='nuclei_'+name;v=dict(prompt=s['prompt'],run_id=s['run_id'],phase=phase,epoch=row['epoch'],split=split,evaluation_contract_id=pm.get('evaluation_contract_id',M),class_name=name,source_sha256=digest)
       for k in ['precision','recall','f1']:v['roi_'+k]=a.get(k+'_by_class',{}).get(key,M);v['pooled_'+k]=b.get(k+'_by_class',{}).get(key,M)
       for k in ['TP','FP','FN']:v['pooled_'+k]=b.get('class_metrics',{}).get(key,{}).get(k,M)
       w.writerow(v);n+=1
 print(json.dumps(dict(group=group,class_rows=n)))
