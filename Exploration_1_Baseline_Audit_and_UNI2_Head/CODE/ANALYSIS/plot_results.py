"""Regenerate the four report figures from preserved measurements."""
import csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
OUT=ROOT.parent/'outputs/figures';OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
recipes=['ce','la','balanced_ce'];labels=['Natural CE','Natural adjusted CE','Balanced sampling + CE']
colors=['#0072B2','#D55E00','#009E73'];markers=['o','s','^']
def read(family,name):return json.loads((ROOT/(family+'_results')/(name+'.json')).read_text())
def finish(fig,name):
    fig.tight_layout();fig.savefig(OUT/(name+'.png'),bbox_inches='tight');fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def decorate(ax,y):ax.set_xlabel('Epoch');ax.set_ylabel(y);ax.grid(alpha=.2)

h=list(csv.DictReader((ROOT/'ingestion/history_compact.csv').open()))
x=[int(r['epoch']) for r in h];fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for key,label,col in [('train_diagnostics.loss','Training','#0072B2'),('val.loss.total','Validation','#D55E00')]:
    ax[0].plot(x,[float(r[key]) for r in h],label=label,color=col,marker='o',ms=3)
for key,label,col in [('val.semantic_final.macro_f1','Semantic Macro-F1','#0072B2'),('val.puma.macro_f1','PUMA ROI macro','#D55E00'),('val.puma_summed.macro_f1','PUMA pooled macro','#009E73')]:
    ax[1].plot(x,[float(r[key]) for r in h],label=label,color=col,marker='o',ms=3)
for a,y in zip(ax,['Reported loss','Reported F1']):decorate(a,y);a.legend(fontsize=8);a.set_xlabel('Historical epoch (zero-based)')
ax[0].set_title('Historical run: diverging validation loss');ax[1].set_title('Different evaluation estimands')
finish(fig,'history')

fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for name,label,col,mark in zip(recipes,labels,colors,markers):
    h=read('cnn',name);x=[r['epoch'] for r in h]
    ax[0].plot(x,[r['objective'] for r in h],label=label,color=col,marker=mark,ms=3)
    ax[1].plot(x,[r['val']['macro_f1'] for r in h],label=label,color=col,marker=mark,ms=3)
for a,y in zip(ax,['Training objective (recipe-specific)','Validation Macro-F1']):decorate(a,y);a.legend(fontsize=8)
ax[0].set_title('Small CNN control: 16,300 training nuclei');ax[1].set_title('4,200 validation nuclei; one seed')
finish(fig,'cnn')

fig,ax=plt.subplots(1,2,figsize=(10,4.1))
for j,(name,label,col,mark) in enumerate(zip(recipes,labels,colors,markers)):
    hs=[read('uni2',name+('' if seed==17 else '_seed'+str(seed))) for seed in [17,29,43]]
    arr=np.array([[r['val']['macro_f1'] for r in h] for h in hs]);x=np.arange(1,11)
    ax[0].plot(x,arr.mean(0),color=col,label=label,marker=mark,ms=3)
    ax[0].fill_between(x,arr.mean(0)-arr.std(0,ddof=1),arr.mean(0)+arr.std(0,ddof=1),color=col,alpha=.12)
    best=max(hs[0],key=lambda r:r['val']['macro_f1'])
    ax[1].bar(np.arange(10)+(j-1)*.25,best['val']['recall'],.25,label=label,color=col)
decorate(ax[0],'Validation Macro-F1');ax[0].legend(fontsize=8)
ax[0].set_title('Frozen UNI2: mean ± SD over three seeds')
ax[1].set_xticks(range(10),['Tum','Lym','Pla','His','Mel','Neu','Str','Epi','End','Apo'],rotation=45)
ax[1].set_ylabel('Recall');ax[1].set_ylim(0,1.05);ax[1].set_title('Selected seed17 checkpoint; enriched validation')
ax[1].grid(axis='y',alpha=.2)
finish(fig,'uni2')

fig,ax=plt.subplots(1,2,figsize=(10,3.8))
for name,col,style in [('frozen','#0072B2','-'),('lora','#D55E00','--')]:
    h=json.loads((ROOT/'lora_results_v2'/(name+'.json')).read_text());x=[r['epoch'] for r in h]
    ax[0].plot(x,[r['objective'] for r in h],label=name,color=col,ls=style,marker='o',ms=4)
    for split,mark in [('train','o'),('val','s')]:
        ax[1].plot(x,[r[split]['macro_f1'] for r in h],label=name+' '+split,color=col,ls=style,marker=mark,ms=5 if name=='frozen' else 3)
for a,y in zip(ax,['Training objective','Macro-F1']):decorate(a,y);a.legend(fontsize=8)
ax[0].set_title('Paired five-epoch actual UNI2 dynamics')
ax[1].set_title('30 train / 20 validation; curves overlap')
finish(fig,'lora')
print('Saved four PNG/PDF figure pairs')
