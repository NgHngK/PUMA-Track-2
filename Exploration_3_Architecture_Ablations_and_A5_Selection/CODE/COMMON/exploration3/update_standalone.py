import shutil,json
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';D=R/'93_final_architecture';archive=R/'00_reference/EXPLORATION2_STANDALONE'
if not archive.exists():shutil.copytree(D,archive,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
model='''import torch
from torch import nn

class CachedClassifier(nn.Module):
    """Selected A5 rank-eight appearance x TierA interaction; no BioMask or LoRA."""
    def __init__(self,biology_dim=16):
        super().__init__()
        if biology_dim!=16:raise ValueError('Selected A5 requires the exact16D TierA schema')
        self.head=nn.Linear(1536,10)
        self.ph=nn.Linear(1536,8,bias=False)
        self.pb=nn.Linear(16,8,bias=False)
        self.branch=nn.Linear(8,10,bias=False)
        nn.init.zeros_(self.branch.weight)
    def forward(self,x):
        if x.shape[-1]!=1552:raise ValueError('Expected LN CLS1536 plus standardized TierA16')
        h=x[:,:1536];b=x[:,1536:]
        return self.head(h)+self.branch(self.ph(h)*self.pb(b))

class FrozenStage2(nn.Module):
    def __init__(self,weights,biology_dim=16):
        super().__init__()
        from uni2 import load_uni2
        self.encoder=load_uni2(weights).eval().requires_grad_(False)
        self.classifier=CachedClassifier(biology_dim)
    def train(self,mode=True):
        super().train(mode);self.encoder.eval();return self
    def forward(self,images,normalized_biology):
        with torch.no_grad():h=nn.functional.layer_norm(self.encoder(images).float(),(1536,))
        return self.classifier(torch.cat((h,normalized_biology),1))
'''
(D/'model.py').write_text(model);cfg=json.loads((D/'config.json').read_text());cfg.update(architecture='A5',interaction_rank=8);(D/'config.json').write_text(json.dumps(cfg,indent=2))
for dest in [R/'00_reference/FINAL_CONFIG.json'] :dest.write_text(json.dumps(cfg,indent=2))
t=(D/'train_eval.py').read_text();t=t.replace("cache=Path(cache);contract=", "\n    if config.get('architecture')!='A5' or config.get('interaction_rank')!=8:raise ValueError('Unsupported final architecture contract')\n    cache=Path(cache);contract=");(D/'train_eval.py').write_text(t)
t=(D/'ARCHITECTURE.md').read_text(encoding='utf-8');start=t.index('## Head, initialization and objective');end=t.index('AdamW:',start)
new='''## Head, initialization and objective

The selected Exploration3 architecture is **A5**, a rank-eight multiplicative interaction. For LayerNorm CLS h and train-standardized TierA b:

`u=P_h h; v=P_b b; z=W_a h+a+W_r(u elementwise-multiplied-by v)`.

P_h is biasless Linear1536→8 (12,288 parameters), P_b is biasless Linear16→8 (128), W_r is biasless Linear8→10 (80). The appearance head contains15,370 parameters. Total trainable parameters are **27,866**, compared with15,530 for historical A1; the increase is12,336. The checkpoint has681,394,176 frozen encoder tensors, giving681,422,042 total model parameters. This is a low-rank feature interaction, **not encoder LoRA**. There is no separate additive A1 biology correction in A5.

Construct the appearance head first for matched seed initialization, then P_h, P_b and W_r. Default PyTorch Linear initialization applies to both projections; W_r is exactly zero. Thus initial predictions exactly match the appearance head, while projection gradients start at zero until W_r moves. There is no fusion gate, extra activation, dropout, affine LayerNorm, BioMask or TierB in the selected model. Multiplication introduces the only new interaction. The tested rank is8; no rank16 or further architecture search was opened.

Use the same canonical ten classes and V17 mapping described below. For training counts n_c, draw N labeled examples with replacement using weight1/n_y; expected class mass is uniform. Optimize ordinary CE `−z_y+logsumexp(z)`, tau0, without additional loss weighting. Unmatched proposals labeled−1 remain in full evaluation but are not semantic training targets. The final trainer checks architecture A5/rank8 and exact16D biology input.

'''
t=t[:start]+new+t[end:];first=t.index('**Frozen');last=t.index('## Inputs',first)
t=t[:first]+'''**Frozen UNI2-h +96-source-pixel crop + appearance-linear head + rank-eight appearance × TierA interaction.** This is the single architecture selected after Exploration3 grouped CV and fixed-final-epoch development confirmation. The former A1 standalone package is preserved under00_reference/EXPLORATION2_STANDALONE for history. The encoder remains entirely frozen and always in evaluation mode.

'''+t[last:];t=t.replace('The completed study demonstrates a small development benefit on 150 GT-centered validation features.','Exploration3 completed234 grouped-CV runs and3 candidate confirmation fits. A5 increased combined internal-CV ROI-F1 by0.005997 and final-epoch reused150-cohort ROI-F1 by0.018823. These are development results; the shared confirmation runner logged intermediate validation metrics, a documented deviation from the planned one-evaluation interface. Only fixed epoch10 determined the comparison.')
(D/'ARCHITECTURE.md').write_text(t,encoding='utf-8')
t=(D/'README.md').read_text(encoding='utf-8');t=t.replace('# Standalone next full-data Stage 2 experiment','# Standalone next full-data Stage 2 experiment — A5')
t+='''
## Exploration3 update

The single current architecture is A5: rank8 multiplicative appearance×TierA residual,27,866 trainable parameters, frozen UNI2. It replaces A1 only after matched capacity/shuffle CV controls and three fixed-epoch development confirmations. No BioMask/TierB or LoRA is deployed. The historical A1 package is archived under00_reference/EXPLORATION2_STANDALONE. Commands above remain valid with the updated config/model. Existing immutable UNI2/TierA input caches need not be recomputed solely because a head changes, provided their extraction contracts match; historical experimental caches still use a different interface from this final package.

VERIFICATION.json now refers to A5 strict parameter loading and prediction parity with the executed confirmation checkpoint. Full Stage1 population validation remains pending. The final full-data schedule remains a prospective maximum20 epochs/patience5; all architecture comparisons used a predeclared fixed10-epoch endpoint. No further architecture expansion is recommended.
'''
(D/'README.md').write_text(t,encoding='utf-8');(D/'parameter_table.csv').write_text('component,total_parameters,trainable_parameters\nUNI2_h,681394176,0\nappearance_linear,15370,15370\nappearance_projection,12288,12288\nbiology_projection,128,128\ninteraction_output,80,80\ncomplete_model,681422042,27866\n')
env=json.loads((D/'ENVIRONMENT.json').read_text());env.update(head_parameters=27866,selected_architecture='A5',interaction_rank=8);(D/'ENVIRONMENT.json').write_text(json.dumps(env,indent=2))
print('Final standalone updated to A5')
