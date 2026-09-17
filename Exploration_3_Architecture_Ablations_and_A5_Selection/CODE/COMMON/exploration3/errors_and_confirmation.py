import json,csv,sys,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from dataset import CLASSES
from train import dump
out=R/'91_error_analysis/exploration3';out.mkdir(exist_ok=True);rows=list(csv.DictReader((R/'01_sample_definition/sample_manifest.csv').open()));ti=np.array([i for i,r in enumerate(rows) if r['split']=='train']);vi=np.array([i for i,r in enumerate(rows) if r['split']=='val']);quality=list(csv.DictReader((R/'00_reference/biomask_audit/quality_per_nucleus.csv').open()));data=np.load(R/'90_comparative_analysis/architecture_oof_logits.npz');y=np.array([int(r['label']) for r in rows]);bios=np.load(R/'02_cache/biology/tierB.npy');masks=np.load(R/'02_cache/biomask/prompt3_gt_diagnostic/masks.npy');rgb=np.load(R/'00_reference/biomask_audit/rgb_crops_qc.npy')
def csvout(p,rs):
    with p.open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
def rv(m):return np.array([np.mean([r.get('nuclei_'+c,{}).get('f1_score',0) for c in CLASSES]) for r in m['puma']['roi_metrics']])
def corr(x,y):
    x=np.array(x);y=np.array(y);q=np.isfinite(x)&np.isfinite(y);return float(np.corrcoef(x[q],y[q])[0,1]) if q.sum()>2 and np.std(x[q])>0 and np.std(y[q])>0 else None
conf=[];deltas=[];recalls=[];allclass=[]
for seed,name in [(17,'exp_fusion_tierA'),(29,'exp_rep_tierA_29'),(43,'exp_rep_tierA_43')]:
    a=json.loads((R/'experiments'/name/'logs/history.jsonl').read_text().splitlines()[-1])['val'];b=json.loads((R/f'experiments/p3_confirm_A5_s{seed}/summary.json').read_text())['selected']['val'];deltas.append(rv(b)-rv(a));recalls.append(np.array(b['recall'])-a['recall'])
    conf.append({'seed':seed,'A1_final_epoch_ROI_F1':a['puma']['fixed10']['macro_f1'],'A5_final_epoch_ROI_F1':b['puma']['fixed10']['macro_f1'],'delta':float((rv(b)-rv(a)).mean()),'A1_semantic_F1':a['macro_f1'],'A5_semantic_F1':b['macro_f1'],'A1_accuracy':a['accuracy'],'A5_accuracy':b['accuracy']})
    for model,v in [('A1',a),('A5',b)]:
        for i,c in enumerate(CLASSES):
            cm=np.array(v['confusion']);p=v['puma']['summed']['class_metrics']['nuclei_'+c];f=v['puma']['fixed10'];fn=cm[i].copy();fn[i]=0;fp=cm[:,i].copy();fp[i]=0
            allclass.append({'model':model,'seed':seed,'class':c,'support':v['support'][i],'positive_ROIs':len({rows[k]['roi'] for k in vi if y[k]==i}),'semantic_P':v['precision'][i],'semantic_R':v['recall'][i],'semantic_F1':v['f1'][i],'one_vs_rest_accuracy':float((len(vi)-fn.sum()-fp.sum())/len(vi)),'global_accuracy':v['accuracy'],'ROI_P':f['precision_by_class']['nuclei_'+c],'ROI_R':f['recall_by_class']['nuclei_'+c],'ROI_F1':f['f1_by_class']['nuclei_'+c],'pooled_P':p['precision'],'pooled_R':p['recall'],'pooled_F1':p['f1_score'],'TP':p['TP'],'FP':p['FP'],'FN':p['FN'],'dominant_FN':CLASSES[fn.argmax()] if fn.sum() else 'none','dominant_FP':CLASSES[fp.argmax()] if fp.sum() else 'none'})
csvout(R/'90_comparative_analysis/prompt3_confirmation.csv',conf);csvout(R/'90_comparative_analysis/prompt3_confirmation_per_class.csv',allclass)
rd=np.mean(deltas,axis=0);dist=rd[np.random.default_rng(17).integers(0,len(rd),(2000,len(rd)))].mean(1);passrule=rd.mean()>=.003 and sum(r['delta']>0 for r in conf)>=2 and np.mean(recalls,axis=0).min()>=-.10
final={'selected':'A5' if passrule else 'A1','confirmation_pass':bool(passrule),'mean_delta':float(rd.mean()),'roi_bootstrap_95':np.quantile(dist,[.025,.975]).tolist(),'class_recall_deltas':np.mean(recalls,axis=0).tolist(),'scope':'reused150 development confirmation; not independent external test','deviation':'shared runner computed/logged intermediate validation metrics at all10 epochs, although comparison used only predeclared final epoch. No checkpoint/architecture reselection used intermediate scores.'};dump(R/'90_comparative_analysis/prompt3_final_decision.json',final)
analyses={};disrows=[];diagmetrics=json.loads((R/'90_comparative_analysis/architecture_oof_metrics.json').read_text())
seedsummary=list(csv.DictReader((R/'90_comparative_analysis/architecture_seed_summary.csv').open()));candidates=[r['architecture'] for r in seedsummary if r['control']=='real' and float(r['paired_delta_vs_A1'])>0]
for kind in candidates:
    pool=[]
    for seed in [17,29,43]:
        p0=data[f'A1_real_{seed}'].argmax(1);p1=data[f'{kind}_real_{seed}'].argmax(1);correct0=p0==y[ti];correct1=p1==y[ti];dv=correct1.astype(int)-correct0.astype(int);norm=np.zeros(len(rows))
        for fold in range(3):
            folder=R/f'experiments/p3_{kind}_real_s{seed}_cv{fold}';s=json.loads((folder/'summary.json').read_text())['selected'];ix=np.load(folder/'predictions.npz')['indices'];norm[ix]=s['diagnostics'].get('correction_norm',np.zeros(len(ix)))
        for j,i in enumerate(ti):
            cat='both_correct' if correct0[j] and correct1[j] else ('A1_wrong_candidate_correct' if correct1[j] else ('A1_correct_candidate_wrong' if correct0[j] else 'both_wrong'))
            q=quality[i];r={'architecture':kind,'seed':seed,'uid':rows[i]['uid'],'class':rows[i]['class_name'],'category':cat,'classification_delta':int(dv[j]),'dice':float(q['dice']),'iou':float(q['iou']),'area_relative_error':float(q['area_relative_error']),'correction_norm':float(norm[i]),'index':int(i)};pool.append(r);disrows.append(r)
    analyses[kind]={'counts':dict(collections.Counter(r['category'] for r in pool)),'dice_vs_delta':corr([r['dice'] for r in pool],[r['classification_delta'] for r in pool]),'area_error_vs_delta':corr([r['area_relative_error'] for r in pool],[r['classification_delta'] for r in pool]),'correction_norm_vs_delta':corr([r['correction_norm'] for r in pool],[r['classification_delta'] for r in pool]),'quality_correction_correlation':corr([r['dice'] for r in pool],[r['correction_norm'] for r in pool]),'mask_quality_strata':{}}
    for good in [True,False]:
        sub=[r for r in pool if (r['dice']>=.8)==good];analyses[kind]['mask_quality_strata']['Dice>=.8' if good else 'Dice<.8']=dict(collections.Counter(r['category'] for r in sub))
    # Deterministic disagreement QC; all four categories, up to2 per category, seed17.
    chosen=[]
    for cat in ['A1_wrong_candidate_correct','A1_correct_candidate_wrong','both_wrong','both_correct']:chosen.extend(sorted([r for r in pool if r['seed']==17 and r['category']==cat],key=lambda r:r['uid'])[:2])
    cols=4 if kind.startswith('B') else 2;fig,axs=plt.subplots(len(chosen),cols,figsize=(cols*3,len(chosen)*2.2),squeeze=False)
    for j,r in enumerate(chosen):
        i=r['index'];axs[j,0].imshow(rgb[i]);axs[j,0].set_title(r['class']+' '+r['category'].replace('_',' '),fontsize=7);axs[j,1].text(.02,.7,r['uid']+'\n'+f"delta={r['classification_delta']}, norm={r['correction_norm']:.3f}",fontsize=7,wrap=True)
        if cols==4:axs[j,2].imshow(masks[i],vmin=0,vmax=1);axs[j,2].set_title(f"Pred mask Dice={r['dice']:.3f}",fontsize=8);axs[j,3].text(.02,.6,f"log soft area={bios[i,0]:.2f}\nmajor={bios[i,1]:.2f}\nminor={bios[i,2]:.2f}\nquality={bios[i,17]:.2f}",fontsize=8)
        for ax in axs[j]:ax.axis('off')
    fig.suptitle(kind+' development disagreements; no causal inference from examples');fig.tight_layout();fig.savefig(out/f'{kind}_disagreements.png',dpi=110,bbox_inches='tight');plt.close(fig)
csvout(out/'disagreement_per_nucleus.csv',disrows);dump(out/'complementarity.json',analyses)
# Summarize gate diagnostics using corresponding fold order and held labels.
gate=[]
for p in sorted((R/'experiments').glob('p3_A[34]_real*/summary.json')):
    s=json.loads(p.read_text());d=s['selected']['diagnostics'];c=s['config'];q=np.load(p.parent/'predictions.npz');rec={'id':s['id'],'architecture':c['architecture'],'seed':c['seed'],'fold':c['fold']}
    if c['architecture']=='A3':rec.update(gates=d['class_gates'],spread=float(np.ptp(d['class_gates'])))
    else:
        g=np.array(d['gate']);correct=q['logits'].argmax(1)==q['labels'];app=np.array(d['appearance_correct_class'])==q['labels'];helped=(~app)&correct;rec.update(mean=float(g.mean()),std=float(g.std()),min=float(g.min()),max=float(g.max()),vs_entropy=corr(g,d['entropy']),vs_correct=corr(g,correct.astype(int)),vs_helped=corr(g,helped.astype(int)))
    gate.append(rec)
dump(out/'gate_diagnostics.json',gate);print(json.dumps(final))
