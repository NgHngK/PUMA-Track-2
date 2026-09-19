from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
F=Path(__file__).resolve().parent;O=F.parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.15,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
S=[17,29,43];colors=['#0072B2','#E69F00','#009E73'];sources=[]
def load(p):
 sources.append(p);return json.loads(p.read_text())
def save(fig,name):
 fig.savefig(F/(name+'.pdf'),bbox_inches='tight');fig.savefig(F/(name+'.png'),dpi=300,bbox_inches='tight');plt.close(fig)
data={}
for n in [300,600,900]:
 for s in S:
  p=O/f'EXPERIMENTS/p6_A_D{n}_s{s}/history.jsonl';sources.append(p);data[n,s]=[json.loads(x) for x in p.read_text().splitlines()]
fig,axes=plt.subplots(2,2,figsize=(10,7.4),layout='constrained')
for n,col in zip([300,600,900],colors):
 vals=np.array([[x['dev']['puma']['fixed10']['macro_f1'] for x in data[n,s]] for s in S]);x=np.arange(1,11)
 axes[0,0].plot(x,vals.mean(0),color=col,label=f'D{n}');axes[0,0].fill_between(x,vals.min(0),vals.max(0),color=col,alpha=.12)
 for split,ls in [('train','--'),('dev','-')]:
  vals=np.array([[h[split]['macro_f1'] for h in data[n,s]] for s in S]);axes[0,1].plot(x,vals.mean(0),ls,color=col,label=f'D{n} {split.upper()}')
axes[0,0].set(title='DEV ROI fixed10 F1: mean and three-seed range',xlabel='Epoch',ylabel='ROI F1',ylim=(0,.2));axes[0,0].legend()
axes[0,1].set(title='Semantic F1: persistent generalization gap',xlabel='Epoch',ylabel='Semantic macro F1',ylim=(0,1));axes[0,1].legend(ncol=2,fontsize=8)
CL=['tumor','lymphocyte','plasma','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']
for n,col,off in [(300,colors[0],-.18),(900,colors[2],.18)]:
 r=np.array([data[n,s][-1]['dev']['recall'] for s in S]);axes[1,0].bar(np.arange(10)+off,r.mean(0),width=.35,color=col,label=f'D{n}')
axes[1,0].set(title='Epoch10 DEV recall: mean across seeds',ylabel='Recall',ylim=(0,1),xticks=np.arange(10),xticklabels=CL);axes[1,0].tick_params(axis='x',rotation=55);axes[1,0].legend()
summ=load(O/'TABLES/DATA_SCALING_SUMMARY.json')
for i,c in enumerate(summ['comparisons']):
 delta=c['paired_roi_delta'];lo,hi=c['roi_ci95'];axes[1,1].errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color=colors[i+1],capsize=4)
axes[1,1].axvline(0,color='gray',lw=1);axes[1,1].set(title='Paired ROI effect versus D300, 95% bootstrap CI',xlabel='Absolute ROI F1 difference',yticks=[0,1],yticklabels=['D600','D900'],ylim=(-.5,1.5));axes[1,1].text(.02,.05,'Both fail the class-recall guard',transform=axes[1,1].transAxes,color='#A33')
fig.suptitle('Stage A: more unique nuclei improve averages but harm neutrophil recall',fontsize=12)
save(fig,'fig_A_scaling')
c=load(O/'TABLES/CACHED_LONG_TAIL_SUMMARY.json');rows=list(csv.DictReader((O/'TABLES/CACHED_LONG_TAIL_CONTROLS.csv').open()));sources.append(O/'TABLES/CACHED_LONG_TAIL_CONTROLS.csv')
arms=['inverse','natural','tempered','crt_reset','crt_noreset'];labels=['Inverse CE','Natural CE','Tempered CE','cRT reset','cRT no-reset']
fig,axes=plt.subplots(1,3,figsize=(11,4.5),layout='constrained')
for i,a in enumerate(arms):
 v=[float(r['roi_f1']) for r in rows if r['arm']==a];axes[0].barh(i,np.mean(v),color='#0072B2' if i==0 else '#B7C4CD',height=.6);axes[0].scatter(v,np.full(3,i),color='#222',s=17,zorder=3)
 axes[2].barh(i,c['means'][a]['gap'],color='#0072B2' if i==0 else '#B7C4CD',height=.6)
for ax in [axes[0],axes[2]]:ax.set_yticks(range(5),labels);ax.invert_yaxis()
axes[0].set(title='Endpoint ROI F1; dots are seeds',xlabel='DEV ROI F1',xlim=(0,.18));axes[2].set(title='Semantic train–DEV gap',xlabel='Macro F1 gap',xlim=(0,.6))
for i,r in enumerate(c['comparisons'][:4]):
 delta=r['roi_delta'];lo,hi=r['ci95'];axes[1].errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color='#D55E00',capsize=4)
axes[1].axvline(0,color='gray',lw=1);axes[1].set(title='Paired effect versus inverse',xlabel='ROI F1 difference, 95% ROI CI',yticks=range(4),yticklabels=labels[1:]);axes[1].invert_yaxis()
fig.suptitle('Stage C cached controls: no promotion; conditional on augmentation study',fontsize=12)
save(fig,'fig_C_cached_controls')
unique=sorted(set(sources));(F/'COMPLETED_CONTROLS_FIGURE_LINEAGE.json').write_text(json.dumps(dict(sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in unique],uncertainty='A curves: three-seed min/max; effect intervals: 2000 paired ROI bootstrap draws after averaging seed effects',holdouts_opened=False),indent=2),encoding='utf-8')
print('Saved two vector PDF figures and 300-DPI PNG previews with source lineage.')
