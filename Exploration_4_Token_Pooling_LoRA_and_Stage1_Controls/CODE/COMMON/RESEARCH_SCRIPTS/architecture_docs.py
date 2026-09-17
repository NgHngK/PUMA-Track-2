import json,csv,collections
from bootstrap import *
from package_history import write_new
DESCRIPTION={
'A0':('Appearance only','z=Wh+a','Frozen CLS1536 → nonaffine LN → Linear1536→10','Remove biology to establish appearance baseline.',15370),
'A1':('Linear TierA correction','z=Wh+a+Vb','CLS1536 + standardizedRGB16 → independentLinear16→10 → sum logits','Test marginal value of direct RGB/point features.',15530),
'A2':('Small nonlinear TierA correction','z=Wh+a+V2 GELU(V1 b+c1)+c2','16→16GELU→10, final weights/bias zero','Test nonlinear biology without large fusion capacity.',15812),
'A3':('Class-gated TierA correction','z=Wh+a+sigmoid(g)⊙Vb','Linear16→10 multiplied by10class scalars initialized0 (.5gates)','Test class-dependent trust in biology; reject if gates remain uniform.',15540),
'A4':('Uncertainty-gated TierA correction','z=Wh+a+sigmoid(w_g·[H(softmax(z_app))/log10,top1−top2]+c_g)Vb','Detached appearance entropy/margin2→1sigmoid gate','Test whether biology is conditionally useful when appearance is uncertain.',15533),
'A5':('Rank8 appearance×TierA interaction','z=Wh+a+Wr[(Ph h)⊙(Pb b)]','1536→8 and16→8, product8→10zero-init','Test cross-feature interactions with low parameter count.',27866),
'A6':('Concatenation bottleneck','z=W2 GELU(W1[h;b]+c1)+c2','concat1552→32GELU→10; replaces appearance-linear head','Test low-dimensional joint bottleneck; cannot share linear appearance initialization.',50026),
'B0':('TierB-only diagnostic','z=Vb+a','Predicted mask biology20→10','Assess standalone mask-feature discrimination; contaminated upstream.',210),
'B1':('TierB morphology diagnostic','z=Wh+a+V b_morph','CLS1536 plus8mask morphology features→10zero-init','Measure morphology contribution; contaminated upstream.',15450),
'B2':('TierB stain/texture diagnostic','z=Wh+a+V b_stain','CLS1536 plus6mask-weighted stain/texture features→10zero-init','Measure mask-weighted stain contribution; contaminated upstream.',15430),
'B3':('Full TierB diagnostic','z=Wh+a+V b_B','CLS1536 plus20mask features→10zero-init','Test full compactmaskfeatures; contaminated upstream.',15570),
'B4':('TierA+TierB diagnostic','z=Wh+a+V_A b_A+V_B b_B','CLS1536 plus36combined biology features→10zero-init','Test complementarity; combined linear36is algebraically separate16+20 corrections.',15730)}
def paragraph_table(rows):return '| Component | Input → output | State | Function / reason | Failure mode |\n|---|---|---|---|---|\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)+'\n'
def historical():
 for n in [1,2,3]:
  for d in (R/f'PROMPT_{n}/ARCHITECTURES').iterdir():
   if not d.is_dir():continue
   arch=d.name
   if n==3 and arch in DESCRIPTION:
    title,eq,flow,why,params=DESCRIPTION[arch];body=f'## Motivation\n\n{why}\n\n## Exact forward\n\n`{eq}`. {flow}. Here h is nonaffine-LN1536, b is the selected feature vector standardized using only current training rows.\n\n## Parameter count\n\n{params:,} trainable head parameters. Encoder frozen; no LoRA. Models and per-run measured counts are in RESULTS/RUNS/<ID>/summary.json.\n\n'
    if arch=='A5':body+='Per component: appearance1536×10+10=15,370; Ph1536×8=12,288; Pb16×8=128; Wr8×10=80. Total27,866. InitialWr=0 gives exact appearance baseline; first-step gradient toPh/Pb is0, whileWr can receive gradient; afterWrupdates projections can learn. This intended delayed gradient is not minority starvation.\n\n'
    if arch=='A2':body+='V1bias included272params;V2bias included170params. ZeroV2/c2 initially suppresses biology; V1learns only afteroutput moves.\n\n'
    if arch=='A3':body+='All10gate logits start0, yielding0.5. The source records eachgate everyepoch. Nearlyuniform gate values failtheprospectiveclass-conditioning criterion.\n\n'
    if arch=='A4':body+='The uncertainty inputs are detached, preventing gate-learning from manipulating appearance entropy directly. Gateweights/bias initialized0. Entropy normalized bylog10.\n\n'
    if arch.startswith('B'):body+='**Validity:** historical BioMask predictions are inference-available, but their upstreamcross-fitting overlapscurrentoutergroups. These runs are contaminated diagnostics and cannot promote a deployable architecture. Maskquality metrics and GTmasks are not inputs. B1usescolumns0:8,B2columns8:14,B3all20,B4TierA16+all20. B3/B4shuffleonlyTierB; TierAstaysrealinB4.\n\n'
   elif n==2:
    title='Frozen-encoder control family '+arch;why='Exploration2 isolates FOV, prior correction, biology and capacity after exact localV17parity.';eq='z=Wh+a (+ Vb when biology is present), or z=Vb+a for biology-only';flow='sourcecrop→frozenUNI2CLS1536→nonaffineLN→linear10; optionalzero-initbiaslessbiologylinear';params='Per-config measured count; varies with selected feature families'
    body=f'## Motivation\n\n{why}\n\n## Exact forward\n\n`{eq}`. {flow}. BIOLOGY_ONLY omits the encoder and consumes only its configured biology columns. The fixed parameter count for appearance-only is15,370; for fullTierA linear correction15,530; biology-onlyfull16is170. Ablation/subfamilycounts depend on exact selectedcolumns. Oracle dimensionality is recorded in its schema/normalizer, not inferred from thename.\n\n'
   elif n==1:
    title=arch;why='Initial real-image mechanisms and verifiedUNI2 feasibility, beforeexactV17continuation.';eq='See exact recovered module and historical recipe';flow='RGB+coordinate→crop→image encoder or cachedUNI2→10logits';params='See source and archived log'
    if arch=='CNN':body='## Motivation and forward\n\nReal-image class-exposure control, not a UNI2surrogate. FOV96 resized48,RGB ImageNetnormalization. ThreeConv3×3 blocks channels3→16→32→64, eachGELU+MaxPool2; adaptiveavg3×3→flatten576→Linear10. WholeCNNtrainable. Training D4rotation/flip; naturalCE,logit-adjustedCEtau1,andbalancedCE compare identicalarchitecture. AdamW.001/WD.01,batch64,10epochs,seed17. See CODE/original_training_local.py.\n\n'
    elif arch=='UNI2_PROBE':body='## Motivation and forward\n\nVerifiedUNI2-h checkpoint,327enriched nuclei (220train/107val). FOV96→resize224→frozenCLS1536→nonaffineLN→Linear1536→10(15,370params). Three losses/samplers andseeds17/29/43. Per-recipe semanticvalidation-best checkpoint; no exactV17metric was reported inthesehistoricalruns. Noencoderadaptation.\n\n'
    elif arch=='LORA_CORRECTED':body='## Motivation and forward\n\nCorrected 50nucleus pilot (30train/20val),5epochs,seed17. CacheUNI2blocks0–19 thenrunblocks20–23 andCLS. Q/Vr8alpha16lateblockLoRA196,608params plus15,370linearhead. SeparateheadLR.001andadapterLR1e-5. Sameinitializedhead andseededloader inbotharms. Frozencontrol andadaptedmodel bothreachtrainF1=1andvalF1=.3038; noobservedclassificationadvantage. This tinypilotdidnotsettleadaptationglobally;Exploration4reopensit.\n\n'
    else:
     body='## Historical exclusion\n\nThis directory preserves an interrupted or unresolved historical run. Exact original executable version is not recovered. Do not attribute these files to a corrected architecture merely because filenames resemble it. Status: UNRESOLVED / NOT EXECUTABLE as originally run. No inventedmetrics orcheckpoints.\n\n';flow='historical artifacts→unresolved provenance→no scientificpromotion'
   else:continue
   comp=[['Coordinates','ROIxy→cropcenter','Stage1immutable / GTcontrol labeled','Selectrequestednucleus; no tissue segmentation','GTvsproposal mismatch'],['Crop','96×96RGB→224×224RGB','fixed','Whitepadding,bicubicresize; exactsourcefunction','Borders,physicalscale shift'],['Encoder','B×3×224×224→B×1536','frozen unless explicitlyLoRA','VerifiedUNI2-hCLS; no registertokens mistakenforpatches','GlobalCLS dilution'],['Feature normalization','1536→1536; bioD→D','LNnonaffine; train-onlystats','Stabilizesfeaturemagnitudes without valstatistics','Nearzero stdclamped1e-6'],['Classifier / correction','h,b→10logits','trainable','Forwardequation above tests namedhypothesis','Overfit,tailcollapse,correctiondominance'],['Evaluation','coords,probabilities,full orsubsetGT→P/R/F1','fixed','ExactsourceV17fromExploration2; historicalP1semantic','Never labelmatchedsubset asfullROI']]
   diagram='''```mermaid
flowchart TD
 I[ROI RGB image] --> C[Coordinate-centered crop]
 X[Fixed coordinate or labeled GT control] --> C
 C --> N[Resize and RGB normalization]
 N --> E[Encoder: frozen unless explicit LoRA]
 E --> H[Representation and nonaffine normalization]
 H --> L[Local architecture-specific classifier]
 I --> B[Configured biology; absent when unused]
 X --> B
 B --> S[Training-only standardization]
 S --> L
 L --> Z[10 logits and probabilities]
 Z --> V[Recorded semantic or exact V17 evaluation]
 X --> V
```
'''
   text=f'# Exploration{n} / {title}\n\n**RECOVERED FROM SOURCE CODE.** This documentation was added inExploration4; it is not presented as originallywritten. Exactconfigs andresults remainunderthisarchitecture.\n\n'+body+'## Component contract\n\n'+paragraph_table(comp)+'\n## Flowchart\n\n'+diagram+'\nThe coordinate controls crop placement. The image encoder produces the appearance vector; only the configured biology path is active. Training-only normalization precedes the exact head equation. Logits become probabilities; their maximum determines semanticconfidence forV17matching. Coordinates remainfixed. ForB0andbiology-only,theencoder/appearancepath is absent; forCNNtheencoderistrainable andcropresize48; forA6theheadisreplaced bytheconcatbottleneck. These explicit exceptions govern thegenericdiagram.\n\n## Training and inference\n\n'+('Exploration3: inverseclassreplacement sampling,ordinaryCE,AdamWLR.001WD.01,batch64,clip1,FP32,seeds17/29/43,10epochs. Eachfoldfitsbiostats ontrainingonly. Fixedepoch10endpoint. Controlsuseappearance-derivedtanhprojection seed1701 ortraining-rowshuffle seed+10000; realvalidationfeaturesremaincorrect forshuffledtraining. Seeexactconfig forwhichbranchisreplaced.\n\n' if n==3 else 'Exactloss,exposure,FOV,seed andcheckpointselection appearin CONFIGS/original andrawhistories. Never substitute a default for an unknown historical fact.\n\n')+'The standalone package reproduces cached-head training locally. Image inputs and the externalUNI2weights are separatelyreferenced; noimplicitnetworkdownload. Missing historical logfields are NOT RECORDED IN ORIGINAL RUN. Runtime is measuredperrun,notestimatedfromparametercount. LocalV17parity result is PROVENANCE/parity_result.json; codecopyhashes are PROVENANCE/code_hashes.json.\n\n## Artifact map\n\n- CODE/: independent localmodules andentrypoints.\n- CONFIGS/original/ andconfig.json: exactrecipes orrecoveredfamilydefaults.\n- RESULTS/: logs,checkpoints,predictions,metrics retainedwithoutconversion.\n- RESULT_INDEX.csv andORIGINAL_PATH_MAP.csv: sourcepaths,hashes andattribution.\n- PROVENANCE/: executablecopychanges,parity andverification.\n'
   write_new(d/'ARCHITECTURE.md',text);write_new(d/'FLOWCHART.md',diagram+'\n'+flow+'\n\nSee ARCHITECTURE.md for component exceptions and equations.\n')
   if not (d/'README.md').exists():write_new(d/'README.md','# Preserved unresolved history\n\nSee ARCHITECTURE.md. Original attribution/executable version isnotestablished; no validtrainingrerun isclaimed.\n')
 print('Historical architecture docs complete',flush=True)
if __name__=='__main__':historical()
