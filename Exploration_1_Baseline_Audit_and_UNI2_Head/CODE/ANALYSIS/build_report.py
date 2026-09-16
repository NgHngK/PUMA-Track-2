import json,csv,sys,shutil,zipfile
from pathlib import Path
ROOT=Path(__file__).parent;OUT=ROOT.parent/'outputs';sys.path.insert(0,str(OUT))
import numpy as np
from dataset import CLASSES

def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(str(x) for x in row)+' |' for row in rows])
def fmt(x):return f'{x:.4f}'

def main():
    body=(ROOT/'report_body.md').read_text(encoding='utf-8')
    audit=json.loads((ROOT/'ingestion/annotation_audit.json').read_text());total=sum(audit['counts'].values())
    ratios=[69.71,13.63,.60,6.84,.78,.24,2.85,2.53,1.75,1.07]
    counts=table(['ID','Class','Local nuclei','Local %','Ratio text %','Positive ROIs'],
        [[i,c,audit['counts']['nuclei_'+c],fmt(100*audit['counts']['nuclei_'+c]/total),ratios[i],audit['coverage']['nuclei_'+c]] for i,c in enumerate(CLASSES)])
    split=json.loads((ROOT/'manifest.json').read_text())['counts']
    splits=table(['Class','Full development train','Full development validation'],[[c,split['train'][i],split['val'][i]] for i,c in enumerate(CLASSES)])
    history=list(csv.DictReader((ROOT/'ingestion/history_compact.csv').open()))
    htable=table(['Epoch (zero-based)','Training loss','Validation loss','Semantic Macro-F1','PUMA ROI macro','PUMA pooled macro'],
                 [[r['epoch']]+[fmt(float(r[k])) for k in ['train_diagnostics.loss','val.loss.total','val.semantic_final.macro_f1','val.puma.macro_f1','val.puma_summed.macro_f1']] for r in history])
    data=json.loads((ROOT/'results_summary.json').read_text());uni=[r for r in data if r['family']=='uni2'];cnn=[r for r in data if r['family']=='cnn'];lora=[r for r in data if r['family']=='lora']
    assert len(uni)==9 and len(cnn)==3 and len(lora)==2,'Wait for all planned probes'
    results='**Design and support.** The CNN uses a 48×48 resized version of the 96-pixel source crop and three small convolutional blocks, not UNI2. It includes 20,500 nuclei from all 205 ROIs. The real UNI2 pilot uses 220 training and 107 validation nuclei; its class-enriched validation spans 38 ROIs. Full feature extraction took 1,121 seconds. Its weights SHA256 is `32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe`.\n\n'
    cs=json.loads((ROOT/'cnn_results/supports.json').read_text());us=json.loads((ROOT/'uni2_features.json').read_text())['counts']
    results+=table(['Class','CNN train','CNN val','UNI2 train','UNI2 val'],[[c,cs['train'][i],cs['val'][i],us['train'][i],us['val'][i]] for i,c in enumerate(CLASSES)])+'\n\n'
    results+='**CNN controls, one seed, ten epochs each.**\n\n'
    results+=table(['Recipe','Best val F1','Epoch','Final val F1','Best balanced accuracy','Best NLL','Best ECE15'],[[r['recipe'],fmt(r['best_macro_f1']),r['best_epoch'],fmt(r['last_macro_f1']),fmt(r['best_balanced_accuracy']),fmt(r['best_nll']),fmt(r['best_ece'])] for r in cnn])+'\n\n'
    results+='At epoch 10, natural CE gives zero recall to plasma_cell, neutrophil and epithelium. The adjusted and balanced-sampler controls have nonzero recall in all ten classes. All classes have positive training-example gradients. This is evidence that zero recall can coexist with an intact gradient path; it is not proof that balanced sampling is universally better. The sample is uniform within each ROI with a 100-nucleus cap, so it is not an exact global-prevalence sample.\n\n'
    results+='**Real frozen UNI2, three seeds, ten epochs per run.**\n\n'
    results+=table(['Recipe / seed','Best val F1','Epoch','Final val F1','Final train F1','Best balanced accuracy','Best ECE15'],[[r['recipe'],fmt(r['best_macro_f1']),r['best_epoch'],fmt(r['last_macro_f1']),fmt(r['final_train_macro']),fmt(r['best_balanced_accuracy']),fmt(r['best_ece'])] for r in uni])+'\n\n'
    aggregates=[]
    for recipe in ['ce','la','balanced_ce']:
        rr=[r for r in uni if r['recipe']==recipe or r['recipe'].startswith(recipe+'_seed')]
        best=np.array([r['best_macro_f1'] for r in rr]);last=np.array([r['last_macro_f1'] for r in rr])
        aggregates.append([recipe,fmt(best.mean())+' ± '+fmt(best.std(ddof=1)),fmt(last.mean())+' ± '+fmt(last.std(ddof=1))])
    results+=table(['Recipe','Best-checkpoint mean ± sample SD','Final-epoch mean ± sample SD'],aggregates)+'\n\n'
    boot=json.loads((ROOT/'bootstrap.json').read_text());lo,hi=boot['roi_bootstrap_percentile95']
    results+=f"The mean paired best-checkpoint LA-minus-CE difference is **{boot['mean_seed_paired_delta']:+.4f}**. The exploratory 2,000-replicate ROI-bootstrap interval is **[{lo:+.4f}, {hi:+.4f}]**. It includes zero. This interval conditions on development-selected checkpoints and an enriched subset; it is not a confirmatory confidence interval for deployment superiority. Use natural adjusted CE as a provisional candidate and retain ordinary CE as the matched reference; do not declare the loss question settled.\n\n"
    selected=[next(r for r in uni if r['recipe']==name) for name in ['ce','la','balanced_ce']]
    results+='**Classwise F1 / recall at seed17 development-selected UNI2 checkpoints.**\n\n'
    results+=table(['Class','Val support','CE F1 / recall','LA F1 / recall','Balanced CE F1 / recall'],[[c,us['val'][i]]+[fmt(r['best_f1'][i])+' / '+fmt(r['best_recall'][i]) for r in selected] for i,c in enumerate(CLASSES)])+'\n\n'
    results+='Apoptosis recall is zero at all three selected seed17 checkpoints, despite positive training head gradients and high training Macro-F1. Histiocyte is also zero for the CE selection. The pilot does not meet the requested all-class generalization/convergence standard. Further encoder adaptation is not automatically the remedy.\n\n'
    discr=json.loads((ROOT/'discrimination.json').read_text())
    results+='**Exploratory alternate-boundary diagnostic after the zero-recall finding.** Training-only cosine centroids and k-nearest neighbours were evaluated without validation-fitted parameters.\n\n'
    results+=table(['Classifier','Val Macro-F1','Apoptosis recall'],[[k,fmt(discr[k]['macro_f1']),fmt(discr[k]['recall'][9])] for k in ['centroid','knn1','knn5']])+'\n\n'
    results+='Similarity/vote scores are not calibrated probabilities. This diagnostic is exploratory, and nearest-neighbour vote ties use the lowest class index. It does not establish a new production winner.\n\n'
    results+='**Actual UNI2 restricted-LoRA dynamics, five epochs.** Thirty training nuclei (three per class) and twenty validation nuclei (two per class) use fixed GT-centred crops and the same original ROI split. Frozen prefix outputs are cached before block20. The corrected comparison uses identical fresh head initialization and an independent seed17 minibatch generator. Preliminary logs from a different-shuffle comparison were preserved but excluded.\n\n'
    for name in ['frozen','lora']:
        h=json.loads((ROOT/('lora_results_v2/'+name+'.json')).read_text())
        results+=f'**{name}:**\n\n'+table(['Epoch','Train objective','Train Macro-F1','Val Macro-F1','Val NLL','Zero-recall val classes'],[[r['epoch'],fmt(r['objective']),fmt(r['train']['macro_f1']),fmt(r['val']['macro_f1']),fmt(r['val']['nll']),', '.join(CLASSES[i] for i,v in enumerate(r['val']['recall']) if v==0) or 'none'] for r in h])+'\n\n'
    results+='This small dynamics check is not a full-data LoRA comparison. Positive adapter gradients, when reported, verify connectivity, not useful representation change. No adaptation advantage is promoted from twenty validation nuclei.\n\n'
    results+='**Complete ten-epoch trajectories (objective / validation Macro-F1).**\n\n'
    for family in ['cnn','uni2']:
        h={name:json.loads((ROOT/(family+'_results')/(name+'.json')).read_text()) for name in ['ce','la','balanced_ce']}
        results+=f'**{family}, seed17:**\n\n'+table(['Epoch','CE objective / F1','LA objective / F1','Balanced CE objective / F1'],[[e+1]+[fmt(h[name][e]['objective'])+' / '+fmt(h[name][e]['val']['macro_f1']) for name in h] for e in range(10)])+'\n\n'
    results+='Training objectives use different effective priors/samplers and must not be ranked as if they were identical risk functions. The archived JSON files include all epochs for all seeds, class exposures, full confusion matrices, unaugmented training metrics and classwise gradient probes.\n\n'
    results+='![Historical learning curve](figures/history.png)\n\n![CNN controls](figures/cnn.png)\n\n![UNI2 seed replications](figures/uni2.png)\n\n![Corrected LoRA dynamics](figures/lora.png)\n'
    check=json.loads((ROOT/'integrity_checks.json').read_text());checks=table(['Check','Outcome'],[[k,'PASS' if v else 'FAIL'] for k,v in check['checks'].items()])
    geometry=json.loads((ROOT/'uni2_results/geometry.json').read_text())
    geo=f"The empirical training-feature covariance effective rank is **{geometry['train_covariance_effective_rank']:.2f}**, with maximum sample rank **{geometry['train_max_rank']}** for 220 training points. Positive class-centroid cosines are high, but this alone is not proof of collapse: a common component and small sample affect cosine geometry. No PCA/whitening was selected from validation or inserted into the production model."
    implementation='\n\n'.join('### '+name+'\n\n```python\n'+(OUT/name).read_text()+'\n```' for name in ['dataset.py','model.py','loss.py','train_eval.py'])
    for key,value in [('COUNTS',counts),('SPLITS',splits),('HISTORY',htable),('RESULTS',results),('CHECKS',checks),('GEOMETRY',geo),('IMPLEMENTATION',implementation)]:body=body.replace('{{'+key+'}}',value)
    assert '{{' not in body
    (OUT/'STAGE2_RESEARCH_REPORT.md').write_text(body,encoding='utf-8')
    shutil.copy(ROOT/'literature/survey.md',OUT/'LITERATURE_SURVEY.md')
    print('report',len(body),'characters')

if __name__=='__main__':main()
