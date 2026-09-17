import sys,json,csv,collections
from pathlib import Path
import numpy as np,torch
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from train import summary_metrics,dump
from dataset import read_manifest,CLASSES
rows=read_manifest(R/'01_sample_definition/sample_manifest.csv');ti=np.array([i for i,r in enumerate(rows) if r['split']=='train']);y=np.array([rows[i]['label'] for i in ti]);rr=[rows[i] for i in ti];roi_names=sorted({r['roi'] for r in rr});out=R/'90_comparative_analysis';summaries=[];by=collections.defaultdict(list)
for p in sorted((R/'experiments').glob('p3_*_cv*/summary.json')):
    s=json.loads(p.read_text());summaries.append(s);c=s['config'];by[(c['architecture'],c['control'],c['seed'])].append((p.parent,s))
assert len(summaries)==234,len(summaries)
def csvout(p,rs):
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def rec(s,m):
    c=s['config'];v=m['puma'];return {'ID':s['id'],'architecture':c['architecture'],'seed':c['seed'],'control':c['control'],'fold':c.get('fold'),'parameters':s['trainable_parameters'],'TierA':c['architecture'] not in ['A0','B0','B1','B2','B3'],'TierB':c['architecture'].startswith('B'),'BioMask_provenance':'upstream-contaminated row-wise OOF' if c['architecture'].startswith('B') else 'none','gate':c['architecture'] if c['architecture'] in ['A3','A4'] else 'none','rank_or_bottleneck':8 if c['architecture']=='A5' else (32 if c['architecture']=='A6' else 0),**{'ROI_'+k:v['fixed10']['macro_'+k] for k in ['precision','recall','f1']},**{'pooled_'+k:v['summed']['macro_'+k] for k in ['precision','recall','f1']},'Accuracy':m['accuracy'],'BalancedAccuracy':m['balanced_accuracy'],'semantic_precision':m['macro_precision'],'semantic_recall':m['macro_recall'],'semantic_f1':m['macro_f1'],'NLL':m['nll'],'ECE15':m['ece15'],'train_f1':s['selected']['train']['macro_f1'],'gap':s['selected']['gap'],'tail_f1':float(np.mean(np.array(m['f1'])[[2,3,4,5,9]])),'zero_recall_classes':','.join(CLASSES[i] for i,r in enumerate(m['recall']) if r==0),'runtime_seconds':s['runtime_seconds'],'decision':'ineligible contamination' if c['architecture'].startswith('B') else 'evaluate prospective rule'}
csvout(out/'architecture_stress_test.csv',[rec(s,s['selected']['val']) for s in summaries])
agg={};tables=[];pc=[]
for key,pairs in by.items():
    assert len(pairs)==3;z=np.zeros((len(rows),10),np.float32);seen=[];diag={};gaps=[]
    for d,s in pairs:
        p=np.load(d/'predictions.npz');z[p['indices']]=p['logits'];seen.extend(p['indices']);gaps.append(s['selected']['gap'])
        for k,v in s['selected']['diagnostics'].items():diag.setdefault(k,[]).extend(v)
    assert sorted(seen)==sorted(ti);m=summary_metrics(rr,y,torch.from_numpy(z[ti]));s=pairs[0][1];r=rec(s,m);r['ID']='_'.join(map(str,key));r['fold']='combined_OOF';r['gap']=float(np.mean(gaps));r['train_f1']=float(np.mean([v['selected']['train']['macro_f1'] for _,v in pairs]));r['runtime_seconds']=sum(v['runtime_seconds'] for _,v in pairs);tables.append(r)
    ag={'metrics':m,'logits':z[ti],'gap':r['gap'],'parameters':r['parameters'],'diagnostics':diag,'summary':r};agg[key]=ag
    for i,c in enumerate(CLASSES):
        cm=np.array(m['confusion']);fn=cm[i].copy();fn[i]=0;fp=cm[:,i].copy();fp[i]=0;v=m['puma']['summed']['class_metrics']['nuclei_'+c];f=m['puma']['fixed10']
        pc.append({'architecture':key[0],'control':key[1],'seed':key[2],'class':c,'support':m['support'][i],'positive_ROIs':len({r['roi'] for r in rr if r['label']==i}),'semantic_P':m['precision'][i],'semantic_R':m['recall'][i],'semantic_F1':m['f1'][i],'one_vs_rest_accuracy':float((len(y)-fn.sum()-fp.sum())/len(y)),'global_conditional_accuracy':m['accuracy'],'dominant_FN':CLASSES[fn.argmax()] if fn.sum() else 'none','dominant_FP':CLASSES[fp.argmax()] if fp.sum() else 'none','TP':v['TP'],'FP':v['FP'],'FN':v['FN'],'pooled_P':v['precision'],'pooled_R':v['recall'],'pooled_F1':v['f1_score'],'ROI_P':f['precision_by_class']['nuclei_'+c],'ROI_R':f['recall_by_class']['nuclei_'+c],'ROI_F1':f['f1_by_class']['nuclei_'+c]})
csvout(out/'architecture_combined_cv.csv',tables)
for r in pc:
    i=CLASSES.index(r['class']);r['recall_delta_vs_A1']=r['semantic_R']-agg[('A1','real',r['seed'])]['metrics']['recall'][i];r['F1_delta_vs_A1']=r['semantic_F1']-agg[('A1','real',r['seed'])]['metrics']['f1'][i]
csvout(out/'architecture_per_class.csv',pc)
def roi_vector(m):return np.array([sum(r.get('nuclei_'+c,{}).get('f1_score',0) for c in CLASSES)/10 for r in m['puma']['roi_metrics']])
families=sorted({(k[0],k[1]) for k in agg});seedtab=[];promotion={};bootstrap={};rng=np.random.default_rng(17);draw=rng.integers(0,len(roi_names),(2000,len(roi_names)))
for kind,control in families:
    aa=[agg[(kind,control,s)] for s in [17,29,43]];base=[agg[('A1','real',s)] for s in [17,29,43]];scores=np.array([a['metrics']['puma']['fixed10']['macro_f1'] for a in aa]);bs=np.array([a['metrics']['puma']['fixed10']['macro_f1'] for a in base]);rd=np.mean([roi_vector(a['metrics'])-roi_vector(b['metrics']) for a,b in zip(aa,base)],axis=0);dist=rd[draw].mean(1);ci=np.quantile(dist,[.025,.975]);delta=scores-bs
    row={'architecture':kind,'control':control,'ROI_F1_mean':float(scores.mean()),'ROI_F1_std':float(scores.std(ddof=1)),'paired_delta_vs_A1':float(delta.mean()),'improved_seeds':int((delta>0).sum()),'roi_bootstrap_low':float(ci[0]),'roi_bootstrap_high':float(ci[1]),'parameters':aa[0]['parameters'],'decision':'diagnostic only' if kind.startswith('B') else 'not promoted'}
    for key in ['Accuracy','BalancedAccuracy','semantic_f1','pooled_f1','gap']:row[key+'_mean']=float(np.mean([a['summary'][key] for a in aa]))
    if control=='real' and kind in ['A2','A3','A4','A5','A6']:
        pd={c:float(np.mean([agg[(kind,c,s)]['metrics']['puma']['fixed10']['macro_f1'] for s in [17,29,43]])) for c in ['placebo','shuffle']}
        rdclass=np.mean([np.array(a['metrics']['recall'])-np.array(b['metrics']['recall']) for a,b in zip(aa,base)],axis=0);gapdelta=np.mean([a['gap']-b['gap'] for a,b in zip(aa,base)]);gates=np.array(aa[0]['diagnostics'].get('class_gates',[])).reshape(-1,10) if kind=='A3' else None
        rule={'mean_gain':delta.mean()>=.003,'two_seeds':(delta>0).sum()>=2,'capacity':scores.mean()-pd['placebo']>=.002,'shuffle':scores.mean()-pd['shuffle']>=.002,'interval':ci[0]>0,'class_recall':rdclass.min()>=-.10,'gap':gapdelta<=.05,'gate_spread':True if gates is None else np.ptp(gates.mean(0))>.05}
        promotion[kind]={'pass':bool(all(rule.values())),'criteria':{k:bool(v) for k,v in rule.items()},'class_recall_deltas':rdclass.tolist(),'gap_delta':float(gapdelta),'mean_delta':float(delta.mean())};row['decision']='eligible for one confirmation' if all(rule.values()) else 'not promoted'
    seedtab.append(row);bootstrap[kind+'_'+control]={'mean_delta':float(delta.mean()),'interval':ci.tolist(),'roi_delta':rd.tolist(),'ROI_order':roi_names,'replicates':2000,'note':'exploratory reused-development CV; no independent patient grouping'}
csvout(out/'architecture_seed_summary.csv',seedtab);dump(out/'architecture_paired_uncertainty.json',bootstrap);dump(out/'architecture_promotion.json',promotion)
passing=[r for r in seedtab if r['decision']=='eligible for one confirmation'];passing.sort(key=lambda r:-r['ROI_F1_mean']);selected='A1'
if passing:
    best=passing[0]['ROI_F1_mean'];selected=min([r for r in passing if best-r['ROI_F1_mean']<=.002],key=lambda r:r['parameters'])['architecture']
dump(out/'prompt3_selection.json',{'cv_selected':selected,'promotion':promotion,'confirmation_required':selected!='A1','BioMask_promotion_eligible':False})
dump(out/'architecture_oof_metrics.json',{str(k):{kk:vv for kk,vv in v.items() if kk!='logits'} for k,v in agg.items()})
np.savez_compressed(out/'architecture_oof_logits.npz',**{f'{k[0]}_{k[1]}_{k[2]}':v['logits'] for k,v in agg.items()})
print(json.dumps({'selected':selected,'promotion':promotion}));print([(r['architecture'],r['control'],round(r['ROI_F1_mean'],5)) for r in seedtab])
