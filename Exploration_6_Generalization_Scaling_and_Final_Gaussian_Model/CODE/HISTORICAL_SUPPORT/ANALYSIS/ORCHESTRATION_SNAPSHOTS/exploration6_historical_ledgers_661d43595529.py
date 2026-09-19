from pathlib import Path
import json,csv,hashlib,re,collections,sys,io
R=Path(r'C:\Users\Hngk\Documents\Codex'); H=R/'PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4'; O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'; EX='--excluded-exploration1' in sys.argv; P1='--exploration1' in sys.argv or EX; T=O/('TABLES/HISTORICAL_PROMPT1_EXCLUDED' if EX else 'TABLES/HISTORICAL_PROMPT1' if P1 else 'TABLES/HISTORICAL');T.mkdir(exist_ok=True)
M='NOT RECORDED IN ORIGINAL RUN'; CL=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def js(x):return json.dumps(x,separators=(',',':'),ensure_ascii=False) if isinstance(x,(dict,list)) else x
def writer(name,fields):
 f=(T/name).open('w',newline='',encoding='utf-8');w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();return f,w
base=['prompt','run_id','seed','fold','epoch','split']
fg,wg=writer('EPOCH_GLOBAL_METRICS.csv',base+['objective','loss','macro_precision','macro_recall','macro_f1','balanced_accuracy_fixed10','accuracy','nll','ece15','roi_macro_precision','roi_macro_recall','roi_macro_f1','pooled_macro_f1','lr','gradients','unique_examples','unique_groups','repeat_fraction','checkpoint_status','early_stopping_counter','source_sha256'])
fp,wp=writer('EPOCH_PER_CLASS_METRICS.csv',base+['class_name','support','predicted','precision','recall','f1','TP','FP','FN','dominant_confusion','class_diagnostics'])
fc,wc=writer('CONFUSION_BY_EPOCH.csv',base+['true_class','predicted_class','count'])
fe,we=writer('EPOCH_SAMPLER_EXPOSURE.csv',base+['class_name','draws','unique_nuclei','unique_groups','repeat_fraction','roi_exposure','bucket_exposure'])
fd,wd=writer('EPOCH_MODEL_DIAGNOSTICS.csv',base+['diagnostics','parameter_norm','gradients','correction_norm','extra_recorded_fields'])
fr,wr=writer('RUN_CONFIG_SNAPSHOTS.csv',['prompt','run_id','seed','fold','config_path','config_sha256','resolved_config','history_path','history_sha256','epochs','status'])
aliases=[]; seen={};canonical=[];counts=collections.Counter(); scalar=['macro_precision','macro_recall','macro_f1','balanced_accuracy_fixed10','accuracy','nll','ece15']
paths=sorted(H.rglob('history.jsonl'),key=lambda p:('RESULTS/RUNS' not in p.as_posix(),str(p)))
if P1:
 paths=[]
 candidates=(H/'EXPLORATION_1/ARCHITECTURES').glob('LORA_EXCLUDED/RESULTS/ORIGINAL/work/lora_results/*.json') if EX else (H/'EXPLORATION_1/ARCHITECTURES').glob('*/RESULTS/ORIGINAL_RUNS/*.json')
 for p in sorted(candidates):
  if '.partial.' in p.name and (not EX or p.with_name(p.name.replace('.partial.','.')).exists()):continue
  value=json.loads(p.read_text(encoding='utf-8-sig'))
  if isinstance(value,list) and value and isinstance(value[0],dict) and 'epoch' in value[0]:paths.append(p)
for path in paths:
 prompt=path.relative_to(H).parts[0].replace('PROMPT_',''); folder=path.parent.parent if path.parent.name=='logs' else path.parent; run=folder.name; digest=sha(path)
 if P1:run=path.relative_to(H).parts[2]+'_'+path.stem
 key=(prompt,run,digest)
 if key in seen:aliases.append(dict(path=str(path),canonical=seen[key],sha256=digest));continue
 seen[key]=str(path)
 cfgp=next((p for parent in [folder,folder.parent,folder.parent.parent] for p in [parent/'config.json',parent/'resolved_config.json'] if p.is_file()),None)
 cfg=json.loads(cfgp.read_text(encoding='utf-8-sig')) if cfgp else {}
 seed=cfg.get('seed',M);fold=cfg.get('fold',cfg.get('cv_fold',M));n=0
 with (io.StringIO('\n'.join(json.dumps(x) for x in json.loads(path.read_text(encoding='utf-8-sig')))) if P1 else path.open(encoding='utf-8-sig')) as fh:
  for line in fh:
   if not line.strip():continue
   h=json.loads(line);n+=1;epoch=h.get('epoch',M); b=dict(prompt=prompt,run_id=run,seed=seed,fold=fold,epoch=epoch)
   for split in ['train','val','dev']:
    m=h.get(split)
    if not isinstance(m,dict):continue
    bb={**b,'split':split};g={**bb,**{k:m.get(k,M) for k in scalar}}
    g.update(objective=h.get('objective',M),loss=m.get('nll',h.get('train_loss' if split=='train' else 'validation_loss',M)),lr=js(h.get('lr',h.get('LRs',M))),gradients=js(h.get('gradients',h.get('positive_target_head_gradient',M))),unique_examples=h.get('unique_UIDs',h.get('unique_examples',M)),unique_groups=h.get('unique_ROIs',h.get('unique_groups',M)),repeat_fraction=h.get('repeat_rate',h.get('repeat_fraction',M)),checkpoint_status=h.get('checkpoint_status',M),early_stopping_counter=h.get('early_stopping_counter',M),source_sha256=digest)
    p=m.get('puma',{});fix=p.get('fixed10',{});g.update({f'roi_{k}':fix.get(k,M) for k in ['macro_precision','macro_recall','macro_f1']});g['pooled_macro_f1']=p.get('summed',{}).get('macro_f1',M);wg.writerow(g)
    cm=m.get('confusion');valid=isinstance(cm,list) and len(cm)==10 and all(len(x)==10 for x in cm)
    for k,name in enumerate(CL):
     row={**bb,'class_name':name}
     for field in ['support','predicted','precision','recall','f1']:
      values=m.get(field);row[field]=values[k] if isinstance(values,list) and len(values)==10 else M
     if valid:
      tp=cm[k][k];row.update(TP=tp,FP=sum(cm[i][k] for i in range(10))-tp,FN=sum(cm[k])-tp,dominant_confusion=js(sorted([(CL[j],cm[k][j]) for j in range(10) if j!=k and cm[k][j]],key=lambda x:-x[1])))
      for j in range(10):wc.writerow({**bb,'true_class':name,'predicted_class':CL[j],'count':cm[k][j]})
     else:row.update(TP=M,FP=M,FN=M,dominant_confusion=M)
     row['class_diagnostics']=js(m.get('class_diagnostics',M));wp.writerow(row)
   exposure=h.get('exposure',[])
   for k,name in enumerate(CL):
    we.writerow({**b,'split':'train','class_name':name,'draws':exposure[k] if isinstance(exposure,list) and len(exposure)==10 else M,'unique_nuclei':h.get('unique_examples_by_class',[M]*10)[k],'unique_groups':M,'repeat_fraction':h.get('repeat_fraction_by_class',[M]*10)[k],'roi_exposure':js(h.get('ROI_exposure',M)),'bucket_exposure':js(h.get('bucket_exposure',M))})
   wd.writerow({**b,'split':'recorded_scope','diagnostics':js(h.get('diagnostics',M)),'parameter_norm':js(h.get('parameter_norm',h.get('parameter_norms',M))),'gradients':js(h.get('gradients',h.get('positive_target_head_gradient',M))),'correction_norm':js(h.get('correction_norm',M)),'extra_recorded_fields':js({k:v for k,v in h.items() if k not in ['train','val','dev','diagnostics','gradients','parameter_norm','parameter_norms','exposure','ROI_exposure','bucket_exposure','correction_norm']})})
 wr.writerow(dict(prompt=prompt,run_id=run,seed=seed,fold=fold,config_path=str(cfgp) if cfgp else M,config_sha256=sha(cfgp) if cfgp else M,resolved_config=js(cfg) if cfgp else M,history_path=str(path),history_sha256=digest,epochs=n,status='RECOVERED_ORIGINAL_EPOCHS'))
 counts[prompt]+=n;canonical.append(dict(prompt=prompt,run_id=run,source=str(path),sha256=digest,epochs=n))
for f in [fg,fp,fc,fe,fd,fr]:f.close()
(T/'HISTORY_SOURCE_ALIASES.json').write_text(json.dumps(aliases,indent=2),encoding='utf-8')
(T/'HISTORY_SOURCES.json').write_text(json.dumps(canonical,indent=2),encoding='utf-8')
summary=dict(canonical_histories=len(canonical),duplicate_copies=len(aliases),epochs_by_prompt=dict(counts),prompt5_new_training_runs=0,exploration1='EXCLUDED / NOT EXECUTABLE AS ORIGINALLY RUN; preserve numbers without efficacy attribution' if EX else 'Recovered canonical original JSON epoch arrays' if P1 else 'Exploration1 original JSON arrays ingested separately',other_training_records='Auxiliary mask records retained in diagnostics; not substituted for ten-class head trials',remaining='Batch histories and additional non-history.jsonl records require inventory check; full reports not yet generated')
(T/'INGESTION_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(js(summary))
