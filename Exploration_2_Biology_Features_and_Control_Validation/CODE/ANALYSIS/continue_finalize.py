import json,csv,shutil,hashlib,platform
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
W=Path(__file__).resolve().parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';D=R/'93_final_architecture';F=R/'92_figures';F.mkdir(exist_ok=True)
classes=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
records=list(csv.DictReader((R/'90_comparative_analysis/master_experiment_table.csv').open()))
summaries={p.parent.parent.name:json.loads(p.read_text()) for p in (R/'experiments').glob('*/metrics/summary.json')}
assert len(summaries)==30
histories={k:[json.loads(x) for x in (R/'experiments'/k/'logs/history.jsonl').read_text().splitlines()] for k in summaries}
assert all(len(h)==10 for h in histories.values())
def table(headers,rows):
    return '\n| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(str(x).replace('|','/') for x in r)+' |\n' for r in rows)+'\n'
def fmt(x):return f'{float(x):.5f}'
def figure(name):
    plt.tight_layout(pad=2)
    plt.savefig(F/(name+'.png'),dpi=180,bbox_inches='tight',pad_inches=.2);plt.savefig(F/(name+'.pdf'),bbox_inches='tight',pad_inches=.2);plt.close()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
colors=['#0072B2','#D55E00','#009E73','#CC79A7']
selected=summaries['exp_fusion_tierA']['selected'];baseline=summaries['exp_loss_balanced_ce']['selected'] if 'exp_loss_balanced_ce' in summaries else None
print('IDs',list(summaries))
def bar(ids,labels,title,name):
    fig,ax=plt.subplots(figsize=(8,4));vals=[summaries[i]['selected']['val']['puma']['fixed10']['macro_f1'] for i in ids]
    ax.bar(labels,vals,color=colors[0]);ax.set_ylabel('V17 fixed10 ROI macro-F1');ax.set_title(title+'\nGT-centered development subset; seed 17');ax.set_ylim(0,max(vals)*1.25)
    for j,v in enumerate(vals):ax.text(j,v+.001,f'{v:.4f}',ha='center',fontsize=9)
    figure(name)
fovids=[next(k for k in summaries if k.startswith('exp_fov') and str(f) in k) for f in [64,96,128]]
bar(fovids,['64 px','96 px','128 px'],'Single-crop field of view (same LA recipe)','fov')
lossids=[next(k for k in summaries if k=='exp_loss_ce'),fovids[1],next(k for k in summaries if k.startswith('exp_loss') and 'balanced' in k)]
baseid=lossids[-1];baseline=summaries[baseid]['selected']
bar(lossids,['Natural CE','Natural LA','Balanced CE'],'Loss / sampling at FOV96','loss')
promotion=json.loads((R/'90_comparative_analysis/biology_promotion.json').read_text())
fig,ax=plt.subplots(figsize=(7,4));x=np.arange(4)
for j,(seed,vals) in enumerate(promotion['seed_scores'].items()):ax.plot(x,[vals[k] for k in ['none','placebo','shuffle','tierA']],marker=['o','s','^'][j],color=colors[j],label='Seed '+seed)
ax.set_xticks(x,['Appearance','Capacity placebo','Shuffled biology','Real biology']);ax.set_ylabel('V17 fixed10 ROI macro-F1');ax.set_title('Matched seed controls (150 validation features / 40 ROIs)');ax.legend();figure('paired_controls')
fig,axs=plt.subplots(1,2,figsize=(11,4))
for name,col in [(baseid,colors[0]),('exp_fusion_tierA',colors[1])]:
    h=histories[name];label='Appearance' if name==baseid else 'Appearance + biology'
    axs[0].plot(range(1,11),[v['train_loss'] for v in h],color=col,linestyle='--',label=label+' train')
    axs[0].plot(range(1,11),[v['validation_loss'] for v in h],color=col,label=label+' val')
    axs[1].plot(range(1,11),[v['val']['puma']['fixed10']['macro_f1'] for v in h],color=col,marker='o',label=label)
axs[0].set_ylabel('Cross-entropy');axs[1].set_ylabel('V17 fixed10 ROI macro-F1')
for ax in axs:ax.set_xlabel('Epoch');ax.legend(fontsize=8)
fig.suptitle('Seed 17: learning and remaining generalization gap');figure('learning')
fig,axs=plt.subplots(2,1,figsize=(11,7),sharex=True);x=np.arange(10)
for ax,key in zip(axs,['f1','recall']):
    ax.bar(x-.18,baseline['val'][key],.36,label='Appearance',color=colors[0]);ax.bar(x+.18,selected['val'][key],.36,label='Appearance + biology',color=colors[1]);ax.set_ylabel('Semantic '+key);ax.set_ylim(0,1);ax.legend(fontsize=8)
axs[-1].set_xticks(x,[c.replace('_',' ') for c in classes],rotation=30,ha='right');fig.suptitle('Conditional semantic performance, seed 17; tail n=12 each');figure('per_class')
# Preserve supporting records once, rather than copying code into each run.
for name in ['literature','ingestion']:
    shutil.copytree(W/name,R/'00_reference'/name,dirs_exist_ok=True)
shutil.copytree(W.parent/'outputs/figures',R/'00_reference/prior_figures',dirs_exist_ok=True)
for name in ['FINAL_SELECTION_PROTOCOL.md','CONTINUATION_PROTOCOL.md','CONTINUATION_AMENDMENTS.md']:
    if (W/name).exists():shutil.copy2(W/name,R/'00_reference'/name)
scripts=R/'00_reference/reproduction_scripts';scripts.mkdir(exist_ok=True)
for p in W.glob('continue_*.py'):shutil.copy2(p,scripts/p.name)
for name in ['v17_parity.py','final_verify.py','build_standalone.py']:
    shutil.copy2(W/name,scripts/name)
# Correct superseded cache-intention language without changing evaluator semantics.
ep=R/'00_reference/v17_evaluation_contract/EVALUATION_CONTRACT.md';t=ep.read_text();t=t.replace('for continuity and verified FOV96 cache reuse','for continuity; fresh caches were extracted because the legacy cache lacked image-content hashes');ep.write_text(t)
profile=json.loads((R/'00_reference/performance_profile/runtime_profile.json').read_text())
state=json.loads((R/'00_reference/RESEARCH_STATE.json').read_text())
perf='# Measured CPU performance\n\nHardware and raw profiling details are in the adjacent JSON files. All encoder timings are eager FP32; cached-head timings exclude one-time encoder extraction. RSS is sampled process RSS, not an isolated tensor-memory allocation measurement.\n'
perf+=table(['FOV','Examples','Wall seconds','CPU seconds','Examples/s','Peak sampled RSS GB'],[[f,v['new_examples'],fmt(v['seconds']),fmt(v['cpu_seconds']),fmt(v['examples_per_second']),fmt(v['peak_sampled_rss_gb'])] for f,v in state['completed_caches'].items()])
perf+='\nSelected CPU settings: eight threads, batch four, zero workers. All inputs resize to 224 square; timing differences across FOVs are not evidence of encoder complexity changes. Biology preprocessing: 30.1256 seconds for 450 nuclei across 181 ROIs. Loader two-worker startup outweighed its benefit. Eager cached-head execution beat the tested compile configuration, which also cost 64.26 seconds initially. No full-data or GPU runtime is claimed.\n'
perf+=table(['Experiment','Head/evaluation wall seconds'],[[r['experiment_id'],fmt(r['runtime_seconds'])] for r in records]);(R/'00_reference/performance_profile/PERFORMANCE_PROFILE.md').write_text(perf)
text=(W/'continuation_synthesis.md').read_text()+'\n\n## Empirical master table: all 30 new runs\n'
text+=table(['Run','Seed','Params','ROI F1','Pooled F1','Semantic accuracy','Semantic F1','Balanced acc.','NLL','ECE15','Epoch','Seconds'],[[r['experiment_id'],r['seed'],r['trainable_params'],*[fmt(r[k]) for k in ['v17_roi_macro_f1','v17_pooled_macro_f1','supplemental_semantic_accuracy','validation_macro_f1','balanced_accuracy','NLL','development_ECE15']],r['selected_epoch'],fmt(r['runtime_seconds'])] for r in records])
text+='\n## Figures\n'
for name,caption in [('fov','FOV screen; no error bars from a single seed.'),('loss','Matched loss/sampling screen at selected FOV.'),('paired_controls','All three seeds; lines connect matched experimental controls, not time.'),('learning','Training and validation loss, and exact ROI-F1, for seed 17.'),('per_class','Conditional semantic scores; consult the V17 TP/FP/FN tables for end-to-end definitions.')]:text+=f'\n![{caption}](92_figures/{name}.png)\n\n{caption}\n'
text+='\n## Exact sample and positive-ROI support\n'
sample=json.loads((R/'01_sample_definition/sample_manifest.json').read_text());sup=sample['support']
text+=table(['ID','Class','Train cells','Train positive ROIs','Val cells','Val positive ROIs'],[[i,c,sup['train'][c]['n'],sup['train'][c]['rois'],sup['val'][c]['n'],sup['val'][c]['rois']] for i,c in enumerate(classes)])
text+='\n## Matched seed promotion evidence\n'+table(['Seed','Appearance ROI F1','Placebo','Shuffle','Real Tier A','Real minus appearance'],[[s,*[fmt(v[k]) for k in ['none','placebo','shuffle','tierA']],fmt(v['tierA']-v['none'])] for s,v in promotion['seed_scores'].items()])
text+='\nMean delta '+fmt(promotion['mean_roi_delta'])+'; 95% paired ROI percentile interval '+str(promotion['roi_bootstrap_95'])+'. Resample 40 ROIs 2,000 times after averaging paired per-ROI differences across three seeds. This is a conditional exploratory interval, not a correction for all model selection.\n'
text+='\n## Detailed run-by-run evidence and dialectic\n\nAll epoch numbers are one-based. Validation is the same fixed enriched development cohort. “Gradient” columns are positive-target head-gradient norms, not encoder gradients. Objective uses each run’s sampling/loss; train and validation loss columns are unadjusted semantic CE. Every displayed per-class value is from that run’s ROI-selected checkpoint.\n'
classrows=list(csv.DictReader((R/'90_comparative_analysis/per_class_results.csv').open()))
for r in records:
    key=r['experiment_id'];s=summaries[key];h=histories[key];v=s['selected']['val'];config=s['config'];cr=[a for a in classrows if a['experiment_id']==key]
    increases=sum(h[i]['objective']>h[i-1]['objective'] for i in range(1,len(h)))
    decision='Control / diagnostic; not selected as a separate full-scale architecture.'
    if key=='exp_fusion_tierA' or key.startswith('exp_rep_tierA'):decision='Supports the selected architecture; repeated seeds measure variability, not alternative designs.'
    if 'oracle' in key:decision='Oracle-only diagnostic; excluded from deployment regardless of score.'
    text+=f'\n### {key}\n\n**Hypothesis:** {r["hypothesis"]}.\n\n**Controlled setup:** same 300/150 feature manifest, fixed encoder cache, canonical labels and V17 contract. Configuration below identifies all changes; counts and FOV-dependent inputs are explicit. Head-only wall time {s["runtime_seconds"]:.4f} seconds; CPU time {s["cpu_seconds"]:.4f} seconds; sampled RSS {s["peak_sampled_rss_gb"]:.4f} GB. Trainable parameters {s["trainable_parameters"]}.\n\n```json\n'+json.dumps(config,indent=2)+'\n```\n'
    text+=f'\n**Reviewer:** A lower objective is insufficient; does the held-out metric improve, and which classes still receive no recall? **Architect:** the objective increased in {increases} of nine epoch transitions; selected epoch {s["selected"]["epoch"]}, best semantic epoch {s["best_semantic_epoch"]}, best pooled epoch {s["best_pooled_epoch"]}. Selected train-minus-validation semantic-F1 gap {s["selected"]["generalization_gap"]:.5f}. Zero-recall classes: {r["zero_recall_classes"] or "none"}. **Decision:** {decision}\n'
    text+=table(['Epoch','Objective','Train CE','Val CE','Train semantic F1','Val semantic F1','Val ROI F1','Val pooled F1','Min positive grad','Min class exposure'],[[e['epoch'],*[fmt(e[k]) for k in ['objective','train_loss','validation_loss']],fmt(e['train']['macro_f1']),fmt(e['val']['macro_f1']),fmt(e['val']['puma']['fixed10']['macro_f1']),fmt(e['val']['puma']['summed']['macro_f1']),fmt(min(e['positive_target_head_gradient'])),min(e['exposure'])] for e in h])
    text+=table(['Class','n','Pred n','Semantic P','Semantic R','Semantic F1','V17 TP','FP','FN','Pooled P','R','F1','ROI P','R','F1','Main FN destination','Main FP source'],[[a['class'],a['semantic_support'],a['predicted'],*[fmt(a[k]) for k in ['semantic_precision','semantic_recall','semantic_f1']],a['v17_TP'],a['v17_FP'],a['v17_FN'],*[fmt(a[k]) for k in ['v17_precision','v17_recall','v17_f1','v17_ROI_precision','v17_ROI_recall','v17_ROI_f1']],a['dominant_FN_destination'],a['dominant_FP_source']] for a in cr])
    text+=f'\nFull evidence: [metrics](experiments/{key}/metrics/summary.json), [all epochs including per-ROI and gradient arrays](experiments/{key}/logs/history.jsonl), [provenance](experiments/{key}/provenance.json).\n'
text+='\n## Selected architecture specification\n\n'+(D/'ARCHITECTURE.md').read_text().replace('# One next full-scale architecture','### Current implementation')
text+='\n## Compute profile\n\n'+perf
text+='\n## Literature synthesis retained and contextualized\n\nThe following survey was verified on 7 September. Its historical last paragraph describing future tests predates this continuation; the FOV, loss and biology controls above now supply that evidence. No new claim of exhaustive systematic coverage is made.\n\n'+(W/'literature/survey.md').read_text()
text+='\n## Prior empirical record (historical appendix)\n\nThe following is preserved evidence from the earlier study, including its then-current recommendation and limited LoRA test. It is not a second current architecture proposal. Where its appearance-only recommendation differs, the current decision above supersedes it based on the newly authorized controlled biology tests. Prior sample metrics are not mixed with the 450-feature comparison.\n\n'
prior=(R/'00_reference/PRIOR_RESEARCH_REPORT.md').read_text();prior=prior.replace('(figures/','(00_reference/prior_figures/')
text+=prior
text+='\n\n## Completion audit\n\nThe 28 completion requirements in the copied master prompt are covered by ingestion/indexes; exact parity; reproducible sample/QC; profiling/caches; FOV/loss/biology/control experiments; all per-class and ROI/pooled/accuracy tables; error analysis; frozen shared code; standalone package; and this integrated report selecting one next full-scale experiment. Optional Tier B, GT-vs-Stage1 and new LoRA were not mechanically run: their artifact or evidence conditions were not met. The complete full-data experiment and SOTA/clinical validation are not claimed complete.\n'
(R/'STAGE2_RESEARCH_REPORT.md').write_text(text,encoding='utf-8')
# Current state points forward without losing the old ledger.
state.update(status='complete',phase='requested bounded continuation complete; full-scale study is the next separate experiment',completed_new_runs=30,last_completed_experiment='all 30 controls and final integrity verification',selected_architecture='frozen UNI2-h FOV96 + nonaffine LN Linear10 + 16-feature zero-init linear TierA correction',next=['one full-scale architecture in 93_final_architecture/config.json; requires verified fixed Stage1 outer-split provenance'],completion_limit='GT-centered development probes; not full-data validation, SOTA or clinical deployment')
state['final_verification']=json.loads((D/'VERIFICATION.json').read_text());(R/'00_reference/RESEARCH_STATE.json').write_text(json.dumps(state,indent=2))
print('REPORT',len(text),'characters',len(summaries),'new runs')

if __name__=='__main__':pass
