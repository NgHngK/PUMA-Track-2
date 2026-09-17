from bootstrap import *
P=R/'EXPLORATION_4'
specs={
'A0':('Frozen CLS reference','h = LN(T[0])',27866,'The exact Exploration3 A5 head and representation; nine saved-logit comparisons have maximum error0.'),
'A1':('Target-containing patch','h = LN(T[9 + 16 floor(v) + floor(u)])',27866,'One final normalized patch token at the transformed coordinate.'),
'A2':('Gaussian target pooling','w_j = exp(-||g_j-p||²/(2·1.5²)) / Σ_k exp(-||g_k-p||²/(2·1.5²)); h = LN(Σ_j w_j T[9+j])',27866,'Fixed Gaussian width9 source pixels; no learned attention.'),
'A3':('Target neighborhood','h = LN(mean{T[9+j]: |j_x-floor(u)|≤1 and |j_y-floor(v)|≤1})',27866,'Nine neighboring patch tokens; edges clipped. Passed internal CV but failed final class-recall confirmation.'),
'A4':('CLS and Gaussian mean','h = LN((LN(T[0]) + LN(Σ_j w_j T[9+j]))/2)',27866,'Equal normalized mean without new trainable pooling parameters.'),
'B1':('Additional FOV64 context','z = z_A5(CLS96,b) + V64 P64 LN(CLS64)',40234,'P64 is1536→8; V64 is8→10 and initially zero. Failed promotion.'),
'B2':('Additional FOV128 context','z = z_A5(CLS96,b) + V128 P128 LN(CLS128)',40234,'Identical extra capacity to B1; failed promotion.'),
'B_duplicate':('Duplicate FOV96 capacity control','z = z_A5(CLS96,b) + V96 P96 LN(CLS96)',40234,'Same additional capacity with no new scale; one shared control avoids redundant reruns.'),
'D1':('ROI-class sampling control','q(i)=1/(K n_{ROI(i),class(i)})',27866,'Uniform nonempty ROI-class bucket followed by uniform nucleus; ordinary CE. Failed promotion.'),
'C_ALIGNMENT':('Detector-held paired coordinate control','h = A3(RGB, GT center or Stage1 center)',27866,'Both Stage2 partitions restricted to historical detector fold0; matched nuclei only. No full-proposal score. Failed promotion.'),
'DIAG_A3LINEAR':('Plain local linear diagnostic','z = W LN(mean target3×3 tokens) + a',15370,'No TierA. Diagnostic-only comparison against reused Exploration3 appearance-only CLS predictions; excluded from selection.'),
'E0_FROZEN':('Five-epoch matched frozen control','z = z_A5(A3,b)',27866,'Same original300/150, seeds and five-epoch endpoint as restricted LoRA.'),
'E1_LORA':('Restricted late Q/V adaptation','W_Q,V effective = W_Q,V + (16/8) B_Q,V A_Q,V',224474,'196608 adapter parameters across blocks20–23, plus27866 head. K, early blocks, MLP, norms frozen. No promotion.'),
'F_CLEAN_MASK_NETWORK':('Clean GT-only mask network','L = .6 valid-pixel BCE + .4 (1-softDice)',None,'Random initialization;294 non-fold0 GT-prompt training rows,156 held fold0; no semantic-label or Stage1-proposal training input.'),
'F_A3':('Clean-cohort A3 reference','h = LN(mean target3×3 tokens)',27866,'Same80/76 clean mask cohort; this score is not comparable to original300/150 confirmation.'),
'F_GAUSSIAN':('Clean-cohort geometric reference','h = LN(Σ_j w_Gaussian,j T[9+j])',27866,'Fixed sigma1.5 patch units; same cohort as mask controls.'),
'F_MASK':('Clean predicted-mask pooling','h = LN(Σ_j w_mask,j T[9+j])',27866,'96² probability mask translated to appearance crop,6×6 averaged into16² weights, mass normalized. Failed promotion.'),
'F_CLS_MASK':('CLS and clean mask mean','h = LN((LN(CLS)+LN(mask pooled tokens))/2)',27866,'Equal mean, same head capacity; failed promotion.'),
'F_SHUFFLE':('Shuffled mask control','h = LN(Σ_j w_permuted-mask,j T[9+j])',27866,'Fixed seed1701 permutation separately inside train and validation; no label-dependent permutation.'),
'F_CLS_SHUFFLE':('CLS and shuffled mask control','h = LN((LN(CLS)+LN(shuffled-mask tokens))/2)',27866,'Matched control for CLS-plus-mask pooling.'),
'FINAL_A3':('Failed final neighborhood confirmation','z = z_A5(A3,b)',27866,'Fixed epoch10 original300/150; all seeds improved ROI F1, but class recall guard failed. Not recommended.'),
'FINAL_RECOMMENDED_A5':('Single retained full-scale architecture','z = W h + a + R[(P_h h) ⊙ (P_b b)]',27866,'Frozen CLS1536; h=nonaffine LN(CLS); b=training-standardized TierA16. Retained after A3 confirmation failure.')}
for d in (P/'ARCHITECTURES').iterdir():
 if d.name not in specs:continue
 title,eq,params,reason=specs[d.name]
 if params is None:params=json.loads((d/'config.json').read_text())['trainable_mask_parameters']
 masknet=d.name=='F_CLEAN_MASK_NETWORK';rep=title.replace('&','and')
 if masknet:diagram='''flowchart TD
 I[Non-fold0 RGB96x96] --> X[Concatenate RGB3 and Gaussian exploration1]
 C[GT point only] --> X
 X --> M[Trainable mask network base32]
 M --> L[Mask logits96x96]
 G[GT mask and valid region] --> B[BCE plus soft Dice]
 L --> B
 L --> H[Fixed epoch10 held-fold0 mask inference]
 H --> Q[Mask quality and downstream pooling controls]
'''
 else:diagram=f'''flowchart TD
 I[ROI RGB and fixed coordinates] --> C[White-padded source crop96; resize224]
 C --> E[UNI2-h patch14, depth24, width1536, registers8]
 E --> T[Final normalized tokens Bx265x1536]
 T --> H[{rep}]
 H --> N[Nonaffine LayerNorm1536]
 N --> A[Appearance Linear1536 to10]
 I --> B[TierA RGB-point features16]
 B --> S[Training-only standardization]
 N --> PH[Projection1536 to8]
 S --> PB[Projection16 to8]
 PH --> P[Elementwise product8]
 PB --> P
 P --> R[Zero-initialized projection8 to10]
 A --> Z[Sum10 logits]
 R --> Z
 Z --> V[Softmax; fixed coordinates; exact local V17]
'''
 if d.name=='DIAG_A3LINEAR':diagram='''flowchart TD
 I[RGB96 crop resized224] --> E[Frozen UNI2-h final patch tokens]
 E --> P[Coordinate-selected 3x3 mean]
 P --> N[Nonaffine LayerNorm1536]
 N --> L[Linear1536 to10]
 L --> V[Semantic and exact V17 metrics]
'''
 text=f'''# {title}

Exploration4 architecture ID: `{d.name}`. **{params:,} trainable parameters** for the executed configuration. {reason}

## Mathematical and tensor contract

`{eq}`

The UNI2-h final feature sequence has265 tokens: CLS at0, eight registers at1–8, and256 spatial tokens at9–264 in row-major order. Each token has1536 channels. The 224-pixel image gives16×16 patches of14 pixels. The coordinate transformation is `left=floor(x-48+.5)`, `top=floor(y-48+.5)`, `u=(x-left)/6`, `v=(y-top)/6`. Tokens are selected after the encoder's final LayerNorm. Registers are never interpreted as spatial tokens. Local tokens remain globally contextualized by self-attention.

Except for the explicitly plain linear diagnostic and mask network, the retained head uses appearance `W:1536→10` with bias, `P_h:1536→8`, `P_b:16→8`, and `R:8→10`, all three projections biasless. Parameters are15370+12288+128+80=27866. `R` starts at zero; initially only its output projection can receive an interaction-branch gradient. This deliberate initialization is not proof of minority gradient starvation. All per-epoch measured gradients are retained.

## Forward flowchart

```mermaid
{diagram}```

## Training and inference

Frozen-head arms use FP32, inverse-class replacement sampling, ordinary CE, AdamW LR.001, weight decay.01, batch64, norm clip1, seeds17/29/43 and fixed epoch10. D1 changes only the sampler. E0 and E1 use fixed epoch5; E1 adds Q/V LoRA r8 alpha16 at adapter LR1e-5, microbatch4 and effective batch64. F mask-network training uses its separate documented loss and fixed ten-epoch recipe. No train-time augmentation was applied to the fixed encoder caches. No fitted temperature was used; temperature1 preserves the recorded confidence contract.

Train-only TierA mean and standard deviation are saved per run, with standard deviation floored at1e-6. The ten class names and the epithelium/endothelium evaluator permutation are defined in local code. Coordinates are never moved by classification. The archived small-sample metrics are conditional development metrics. Only the full-data evaluator in the final package can score complete proposals against complete ROI annotations, including empty proposal ROIs.

## Controls, provenance and execution

Broken-coordinate A controls use fixed uniform grid centers in[2,14]², independent of labels. Literal shuffling of almost-identical centered coordinates is a weak control and was retained only as a diagnostic. Multiscale controls add an identical rank-eight branch using duplicate96 features. Mask controls use independent mask-shape permutations with fixed seed1701. The exact configuration, normalizer, checkpoint, epoch history and saved predictions for each run are under RESULTS and indexed in RESULT_INDEX.csv. Missing original measurements are not reconstructed as observations.

CODE contains local runtime modules. Cache references identify shared immutable inputs, not code imported from another architecture. README gives the entrypoint commands. PROVENANCE contains source hashes, evaluator parity and execution checks. All recommendations are governed by TABLES/final_decision.json; historical or internal-CV successes do not override failed confirmation.
'''
 for name,body in [('ARCHITECTURE.md',text),('FLOWCHART.md','```mermaid\n'+diagram+'```\n\n'+reason+'\n')]:
  path=d/name
  if path.exists():cp(path,d/'PROVENANCE'/(name+'.initial_draft.txt'))
  path.write_text(body,encoding='utf8')
print('P4 architecture documents',len(specs))
