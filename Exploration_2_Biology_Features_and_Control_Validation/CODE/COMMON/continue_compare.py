import sys,json,csv,collections
from pathlib import Path
import numpy as np
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from dataset import CLASSES,read_manifest
def main():
    records=[];classrows=[];summaries=[]
    for p in sorted((R/'experiments').glob('*/metrics/summary.json')):
        s=json.loads(p.read_text());summaries.append(s);c=s['config'];sel=s['selected'];v=sel['val'];f=v['puma']['fixed10'];pool=v['puma']['summed'];bio=c.get('biology','none')
        rec={'experiment_id':s['id'],'hypothesis':c.get('hypothesis'),'sample_id':'expanded450_seed17','fov':c.get('fov',96),'coordinate_source':'GT area centroid; component-aware V17 GT',
         'encoder_mode':'biology_only' if c.get('biology_only') else 'frozen UNI2','biology_tier':'C_ORACLE' if bio=='oracle' else ('A' if bio in ['tierA','shuffle'] else 'none'),
         'biology_family':','.join(c.get('families',[])) or bio,'fusion':'none' if bio=='none' or c.get('biology_only') else 'zero-init additive linear logits','tau':c.get('tau',1),'sampler':c.get('sampler','natural'),
         'trainable_params':s['trainable_parameters'],'seed':c.get('seed',17),'lr':c.get('lr',.001),'weight_decay':c.get('weight_decay',.01),'runtime_seconds':s['runtime_seconds'],
         'v17_roi_macro_precision':f['macro_precision'],'v17_roi_macro_recall':f['macro_recall'],'v17_roi_macro_f1':f['macro_f1'],
         'v17_pooled_macro_precision':pool['macro_precision'],'v17_pooled_macro_recall':pool['macro_recall'],'v17_pooled_macro_f1':pool['macro_f1'],
         'supplemental_semantic_accuracy':v['accuracy'],'balanced_accuracy':v['balanced_accuracy'],'NLL':v['nll'],'development_ECE15':v['ece15'],
         'train_macro_f1':sel['train']['macro_f1'],'validation_macro_f1':v['macro_f1'],'generalization_gap':sel['generalization_gap'],
         'tail_mean_f1':float(np.mean([v['f1'][i] for i in [2,3,4,5,9]])),'zero_recall_classes':','.join(CLASSES[i] for i,r in enumerate(v['recall']) if r==0),
         'selected_epoch':sel['epoch'],'best_semantic_epoch':s['best_semantic_epoch'],'best_pooled_epoch':s['best_pooled_epoch'],'decision':'exploratory; see final synthesis'}
        records.append(rec);cm=np.array(v['confusion'])
        for i,name in enumerate(CLASSES):
            fn=cm[i].copy();fn[i]=0;fp=cm[:,i].copy();fp[i]=0;pm=pool['class_metrics']['nuclei_'+name]
            classrows.append({'experiment_id':s['id'],'class_id':i,'class':name,'semantic_support':v['support'][i],'predicted':v['predicted'][i],
              'semantic_precision':v['precision'][i],'semantic_recall':v['recall'][i],'semantic_f1':v['f1'][i],
              'dominant_FN_destination':CLASSES[int(fn.argmax())] if fn.sum() else 'none','dominant_FP_source':CLASSES[int(fp.argmax())] if fp.sum() else 'none',
              'v17_TP':pm['TP'],'v17_FP':pm['FP'],'v17_FN':pm['FN'],'v17_GT_support':pm['TP']+pm['FN'],'v17_precision':pm['precision'],'v17_recall':pm['recall'],'v17_f1':pm['f1_score'],
              'v17_ROI_precision':f['precision_by_class']['nuclei_'+name],'v17_ROI_recall':f['recall_by_class']['nuclei_'+name],'v17_ROI_f1':f['f1_by_class']['nuclei_'+name]})
    out=R/'90_comparative_analysis';out.mkdir(exist_ok=True)
    for name,rs in [('master_experiment_table.csv',records),('per_class_results.csv',classrows)]:
        with (out/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    (out/'all_summaries.json').write_text(json.dumps(summaries,indent=2))
    for dest in [R/'experiments/registry.csv',R/'00_reference/EXPERIMENT_INDEX_NEW.csv']:
        with dest.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=['experiment_id','hypothesis','fov','biology_tier','seed','runtime_seconds'],extrasaction='ignore');w.writeheader();w.writerows(records)
    print([(r['experiment_id'],round(r['v17_roi_macro_f1'],4),round(r['validation_macro_f1'],4)) for r in records])
if __name__=='__main__':main()
