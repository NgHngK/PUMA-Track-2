from pathlib import Path
import csv,json,collections
R=Path(r'C:\Users\Hngk\Documents\Codex');S=R/'PUMA_STAGE2_PROMPT6/DATA_PRESELECTION/SELECTION';O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
out=[]
for name in ['D300','D600','D900','LOCKED_NATURAL']:
 with (S/(name+'.csv')).open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
 counts=collections.defaultdict(lambda:[0,0,0])
 for r in rows:
  k=(r['split'],r['class_name']);n=len(json.loads(r['eval_centroids'])) if r['eval_centroids'] else 1;counts[k][0]+=1;counts[k][1]+=n;counts[k][2]+=n>1
 for split in sorted({r['split'] for r in rows}):
  for cl in ['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']:
   v=counts[split,cl];out.append(dict(manifest=name,split=split,class_name=cl,semantic_annotation_UIDs=v[0],V17_GT_components=v[1],multi_component_annotations=v[2]))
with (O/'DATASET_X3/ANNOTATION_VS_COMPONENT_CENSUS.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
print(json.dumps([dict(manifest=name,split=split,annotations=sum(r['semantic_annotation_UIDs'] for r in out if r['manifest']==name and r['split']==split),components=sum(r['V17_GT_components'] for r in out if r['manifest']==name and r['split']==split)) for name,split in [('D900','train'),('D900','dev'),('D900','locked'),('LOCKED_NATURAL','locked_natural')]]))
