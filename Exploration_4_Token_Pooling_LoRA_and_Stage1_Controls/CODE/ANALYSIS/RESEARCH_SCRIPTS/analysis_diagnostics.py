import sys,json,csv,collections
from bootstrap import *
from engine import load_data,norm,extra_metrics,CLASSES,P
from compare import roi_vector
import numpy as np,torch
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dataset import centered_crop
def run():
 rows,cv,y=load_data();ti=np.flatnonzero(cv>=0);D=P/'TARGET_POOLING/DIAGNOSTICS';D.mkdir(exist_ok=True);reps={'CLS':np.load(OLD/'02_cache/uni2_features/gt_fov96/features.npy')};reps.update({k:np.load(P/'TARGET_POOLING/REPRESENTATIONS'/(k+'.npy')) for k in ['A1','A2','A3','A4']});geometry=[]
 cls=norm(reps['CLS']).numpy();cls/=np.linalg.norm(cls,axis=1,keepdims=True)
 for name,raw in reps.items():
  h=norm(raw).numpy();h/=np.linalg.norm(h,axis=1,keepdims=True);sim=h@h.T
  for fold in range(3):
   tr=np.flatnonzero((cv>=0)&(cv!=fold));va=np.flatnonzero(cv==fold);means=np.stack([h[tr][y[tr].numpy()==c].mean(0) for c in range(10)]);mu=h[tr].mean(0);within=float(np.mean(((h[tr]-means[y[tr]])**2).sum(1)));between=float(np.mean(((means-mu)**2).sum(1)));ss=sim[np.ix_(tr,tr)];same=y[tr,None].numpy()==y[tr].numpy()[None];diffroi=np.array([rows[i]['roi'] for i in tr])[:,None]!=np.array([rows[i]['roi'] for i in tr])[None];nnix=np.argsort(sim[np.ix_(va,tr)],axis=1)[:,-5:];purity=(y[tr[nnix]].numpy()==y[va,None].numpy()).mean(1)
   geometry.append({'representation':name,'fold':fold,'mean_cosine_to_CLS':float((h[va]*cls[va]).sum(1).mean()),'train_withinclass_cosine_otherROI':float(ss[same&diffroi].mean()),'train_betweenclass_cosine_otherROI':float(ss[(~same)&diffroi].mean()),'centroid_scatter_ratio':between/max(within,1e-12),'held5NNpurity':float(purity.mean())})
 csvout(D/'representation_geometry.csv',geometry)
 z=np.load(P/'TABLES/A_complete/oof_logits.npz');base=z['A0_17'].argmax(1);new=z['A3_17'].argmax(1);yy=y[ti].numpy();categories={'both_correct':(base==yy)&(new==yy),'rescued':(base!=yy)&(new==yy),'harmed':(base==yy)&(new!=yy),'both_wrong':(base!=yy)&(new!=yy)};selected=[]
 for cat,mask in categories.items():
  ids=np.flatnonzero(mask);tail=[i for i in ids if yy[i] in [2,3,4,5,9]];chosen=(tail+[i for i in ids if i not in tail])[:2]
  selected.extend((cat,int(ti[i]),int(base[i]),int(new[i])) for i in chosen)
 weights=np.load(P/'TARGET_POOLING/REPRESENTATIONS/pooling_geometry.npz');fig,axs=plt.subplots(len(selected),5,figsize=(13,2.5*len(selected)))
 qc=[]
 for j,(cat,i,b,n) in enumerate(selected):
  r=rows[i]
  with Image.open(r['image']) as im:
   for k,fov in enumerate([64,96,128]):axs[j,k].imshow(centered_crop(im.convert('RGB'),r['x'],r['y'],fov));axs[j,k].set_title(f'{cat}: FOV{fov}' if k==0 else f'FOV{fov}')
   crop=centered_crop(im.convert('RGB'),r['x'],r['y'],96)
  axs[j,3].imshow(crop,extent=[0,16,16,0]);axs[j,3].set_xticks(range(17));axs[j,3].set_yticks(range(17));axs[j,3].grid(alpha=.25);xy=weights['coordinates'][i];axs[j,3].scatter(*xy,c='red',s=15);cx,cy=np.floor(xy).astype(int);axs[j,3].add_patch(plt.Rectangle((cx-1,cy-1),3,3,fill=False,edgecolor='yellow',lw=2));axs[j,3].set_title('Target / 3x3 neighborhood')
  axs[j,4].imshow(weights['real'][i].reshape(16,16),cmap='magma');axs[j,4].set_title('Gaussian comparator weights');axs[j,0].set_ylabel(f"GT {CLASSES[r['label']]}\nCLS {CLASSES[b]}\nlocal {CLASSES[n]}",fontsize=8)
  for ax in axs[j]:ax.tick_params(labelbottom=False,labelleft=False,bottom=False,left=False)
  qc.append({'category':cat,'index':i,'uid':r['uid'],'class':r['class_name'],'baseline':CLASSES[b],'local':CLASSES[n]})
 fig.tight_layout();fig.savefig(P/'FIGURES/target_multiscale_QC.png',dpi=150);plt.close(fig);dump(D/'QC_selection.json',qc)
 # Clean historicalfold0 paired alignment endpoint, with separate training arms.
 paired={};ct=[];cmetrics={}
 for kind in ['C_GTTRAIN','C_STAGE1TRAIN']:
  paired[kind]=[]
  for seed in [17,29,43]:
   rec=json.loads((P/f'RUNS/{kind}_s{seed}/paired_evaluation.jsonl').read_text().splitlines()[-1]);paired[kind].append(rec)
   for center in ['GT_centered','Stage1_centered']:
    m=rec[center];ct.append({'training':kind,'seed':seed,'eval_coordinates':center,'ROI_F1':m['puma']['fixed10']['macro_f1'],'semantic_F1':m['macro_f1'],'accuracy':m['accuracy'],'NLL':m['nll'],'ECE15':m['ece15'],'scope':'matched conditional detector-heldfold0'})
 csvout(P/'STAGE1_PROPOSALS/alignment_endpoints.csv',ct)
 aa=paired['C_STAGE1TRAIN'];bb=paired['C_GTTRAIN'];rd=np.mean([roi_vector(a['Stage1_centered'])-roi_vector(b['Stage1_centered']) for a,b in zip(aa,bb)],0);rng=np.random.default_rng(1701);ci=np.quantile(rd[rng.integers(0,len(rd),(5000,len(rd)))].mean(1),[.025,.975]);delta=[a['Stage1_centered']['puma']['fixed10']['macro_f1']-b['Stage1_centered']['puma']['fixed10']['macro_f1'] for a,b in zip(aa,bb)];rc=np.mean([np.array(a['Stage1_centered']['recall'])-np.array(b['Stage1_centered']['recall']) for a,b in zip(aa,bb)],0)
 dump(P/'STAGE1_PROPOSALS/alignment_decision.json',{'seed_deltas':delta,'mean_delta':float(np.mean(delta)),'ROI_bootstrap95':ci.tolist(),'class_recall_deltas':rc.tolist(),'promotion':bool(np.mean(delta)>=.003 and sum(d>0 for d in delta)>=2 and ci[0]>0 and rc.min()>=-.10),'limits':'samefold0 upstream-excluded conditional matched nuclei; not fullproposal score'})
 cohort=json.loads((P/'STAGE1_PROPOSALS/clean_cohort.json').read_text());pairs=json.loads((P/'STAGE1_PROPOSALS/paired450.json').read_text());lookup={p['index']:p for p in pairs};bins=[];events=[]
 for kind in ['C_GTTRAIN','C_STAGE1TRAIN']:
  for seed in [17,29,43]:
   pr=np.load(P/f'RUNS/{kind}_s{seed}/paired_epoch_10.npz');pred=pr['stage1_logits'].argmax(1)
   for j,ci in enumerate(pr['indices']):
    oi=cohort['original_indices'][ci];d=lookup[oi]['displacement_from_area_centroid'];binname=['0–2','2–5','5–10','10–15','>15'][np.searchsorted([2,5,10,15],d,side='right')];events.append({'training':kind,'seed':seed,'uid':rows[oi]['uid'],'class':rows[oi]['class_name'],'displacement':d,'bin':binname,'correct':bool(pred[j]==pr['labels'][j]),'prediction':CLASSES[pred[j]]})
 csvout(P/'STAGE1_PROPOSALS/displacement_predictions.csv',events)
 for kind in ['C_GTTRAIN','C_STAGE1TRAIN']:
  for c in CLASSES+['ALL']:
   for b in ['0–2','2–5','5–10','10–15','>15']:
    ev=[v for v in events if v['training']==kind and (v['class']==c or c=='ALL') and v['bin']==b];bins.append({'training':kind,'class':c,'bin':b,'seed_events':len(ev),'unique_nuclei':len({v['uid'] for v in ev}),'conditional_accuracy':float(np.mean([v['correct'] for v in ev])) if ev else None})
 csvout(P/'STAGE1_PROPOSALS/displacement_bins.csv',bins)
 print('DIAGNOSTICS_COMPLETE; alignment',np.mean(delta),ci,rc,flush=True)
if __name__=='__main__':run()
