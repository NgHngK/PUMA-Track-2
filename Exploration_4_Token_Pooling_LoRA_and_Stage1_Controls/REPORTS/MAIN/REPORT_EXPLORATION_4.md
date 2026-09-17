# Exploration 4: representation study, failed confirmation, and retained architecture

The architecture for the next full-scale experiment remains **Exploration 3 A5: frozen UNI2-h CLS features from one 96-source-pixel crop, nonaffine LayerNorm, a ten-class linear appearance head, and a rank-eight interaction with 16 training-standardized RGB/point features**. Exploration 4 did not provide enough evidence to add a new component. Target-neighborhood pooling improved internal ROI-grouped cross-validation, but it failed the predeclared per-class recall requirement in the reused 150-nucleus confirmation cohort. This means the candidate was not confirmed; it does not mean that CLS is always better.

## Scientific scope and continuity

I kept Stage 1 detection, localization, decoder thresholds, and proposal coordinates fixed. The supplied Stage 1 precision, recall, and F1 values are background information rather than remeasured classification evidence. I did not introduce tissue segmentation. This remains a ten-class nuclei-classification study with GT-centered development controls and a separately identified detector-held matched-proposal comparison. I did not run full-scale classifier training, a complete-proposal encoder pass, external patient validation, clinical validation, or a SOTA evaluation.

The census contains 97,193 GeoJSON annotation features across 205 ROIs. The class counts are: tumor 57,234; lymphocyte 21,643; plasma cell 520; histiocyte 7,168; melanophage 695; neutrophil 366; stroma 3,856; epithelium 2,211; endothelium 1,696; and apoptosis 1,804. The ratio between the largest and smallest classes is 156.38. The supplied ratio file describes a different population, so I did not use it in place of this census. The source arrays contain 97,378 components because polygons and components are represented differently; this does not change the class census. All training corrections use the counts from the current training partition.

Exploration 1 established real-image training feasibility, corrected a misidentified1024-dimensional/16-pixel-patch checkpoint, and verified the1536-dimensional/14-pixel-patch UNI2-h checkpoint. Its semantic metrics did not have exact localV17 parity. Exploration 2 established evaluator parity, controlled FOV and loss, and selected a small linear TierA correction. Exploration 3 expanded the finite head/control study and selected A5, while identifying upstream BioMask contamination. Exploration 4 explicitly reopened representation, multiscale, proposal-coordinate, ROI-class sampling and restricted adaptation questions. All original files remain separate historical evidence. Earlier recommendations are reproduced with their original scope; they are not simultaneous current recommendations.

## Evaluator parity precedes comparisons

The supplied localVersion17 source is authoritative for this archive. The package tests cover six named edge fixtures,100 random fixtures, centroid computation, score-prioritized matching, strict distance boundary, stable ties, collision behavior, dataset aggregation, zero-proposal cases and canonical class adaptation. Canonical classes7/8 are epithelium/endothelium; V17 indices are8/7. The adapter is explicitly tested. This establishes parity with the supplied implementation, not independent certification against a current challenge server.

Nine Exploration 4 A0 recomputations reproduced the corresponding Exploration 3 A5 final logits with maximum absolute error0. These recomputations were the required baseline admission check. Other historical experiments were copied and re-evaluated or checked by forward inference; they were not retrained for archival convenience. Conditional semantic metrics, summed matching metrics and mean fixed-ten-class ROI metrics remain distinct. The subset contains many ROIs without all ten classes, so its fixed-ten ROI score is far below semantic Macro-F1; these quantities must not be interchanged.

## Predeclared protocol and statistical limits

The original 450-row manifest contains 300 training and 150 reused development-confirmation nuclei. Internal comparisons combine the saved three ROI-grouped validation folds across the300 training rows, separately for seeds17,29,43. Endpoints are epoch10, without best-epoch selection. Promotion required mean paired ROI Macro-F1 gain≥.003, positive gains in at least two seeds, a positive paired ROI-bootstrap95% lower bound, advantage≥.002 over the matched negative/capacity control, no mean class recall reduction>.10, and no semantic train–validation gap increase>.05. Bootstrap uses 5000 resamples and RNG 1701; seeds are averaged within ROI before resampling. These are exploratory ROI intervals, not patient-level confidence intervals. Patient identifiers were unavailable; case_id equaled ROI identifier.

Final confirmation used the prespecified fixed epoch 10, a mean gain of at least .003, at least two positive seeds, and no class-recall drop above .10. When the candidate failed, I did not switch to a runner-up. The 150 rows are reused development data, including the LoRA experiment, rather than an untouched test set. I logged every epoch for diagnostics. Here, “convergence” only means that the planned study was completed; it does not prove that the model is globally optimal.

## Reviewer–architect record: executed hypotheses and decisions

### A. Does CLS dilute nucleus-specific information?

**Reviewer:** A pathology foundation model pretrained on tissue patches may encode useful context in CLS while mixing signals from neighboring cells. A classifier-head plateau does not establish an encoder limitation. Show a target-conditioned representation with identical head capacity, then destroy target alignment without changing capacity.

**Architect:** Audit the actual token sequence before pooling. UNI2-h emits CLS0, registers1–8 and spatial tokens9–264. The16×16 spatial grid is row-major, width1536, after final encoder normalization. For the96-source-pixel crop, source crop origin is floor(center−48+.5) and each patch corresponds to6 source pixels. Compare the target-containing token, fixed Gaussian pooling with sigma1.5 patches, a3×3 target neighborhood, and an equal normalized CLS/Gaussian mean. Keep A5 and TierA unchanged.

**Probe evidence:** A3 neighborhood mean ROI F1 = .108758 versus A0=.084098, delta=.024661, interval[.013821,.036091], all three seeds positive. Broken-coordinate A3=.058558. A1=.101731, A2=.106654 and A4=.104686 also passed internal rules. A3 won the declared rule; no learned pooling parameters were added. A2 was outside the.002 parsimony band by.002104. A separately predeclared plain linear diagnostic improved mean ROI F1 by.032785 without TierA, interval[.021797,.043743]. This supports representation value independent of the interaction branch, but does not establish a causal biological mechanism or select a new head.

**Reviewer:** Shuffling coordinates among centered crops can be a meaningless control because all points remain near the center. Local tokens have global attention context; do not call them isolated morphology.

**Remediation:** Record literal-coordinate shuffle displacement and fixed off-center diagnostics; train the strong independent broken-center control uniformly in[2,14]². Interpret A3 as a target-indexed, globally contextualized representation. The8-case QC panel displays correct/correct, rescued, harmed and wrong/wrong examples at64/96/128 FOV, with token neighborhood and Gaussian weights. It is deterministic error inspection, not expert clinical adjudication.

### B. Does extra scale earn its computation?

**Reviewer:** Adding context and parameters together confounds scale with capacity.

**Architect:** Add one1536→8→10 residual, zero-initialized at its output, to the original CLS96 A5. Compare CLS64 or CLS128 against an identical duplicate-CLS96 residual. Each model has40,234 parameters,12,368 more than A5. Keep the first scale tests CLS-based to isolate scale from target pooling.

**Evidence and decision:** FOV64 residual mean ROI F1 = .084334, delta=.000236; FOV128=.085729, delta=.001631; duplicate96=.084389. Neither gain reaches.003 or establishes sufficient advantage over capacity control. Three-scale extension and combinations were not justified and were not run. This rules out these finite residuals, not all physical-scale choices.

### C. Can proposal alignment be studied without upstream leakage?

**Reviewer:** Row-level out-of-fold Stage 1 proposals are not sufficient if the Stage 2 training and validation groups overlap different Stage 1 training folds. Recover the full proposal source and its checkpoint lineage before interpreting adaptation.

**Architect:** Recover109,637 fixed proposals:94,345 matched and15,292 unmatched. Reconcile file SHA, raw-array SHA and typed-array SHA according to their actual source definitions; all agree with their corresponding contracts. The apparent hash discrepancy was a convention discrepancy, not evidence of corruption. Recover the fixed epoch150 detectorfold0 EMA checkpoint and source exclusion logic. Both Stage 2 train and validation are restricted to historical detectorfold0, which that detector excluded entirely. Do not use the unrelated older128,927-proposal cache.

**Evidence:**433 of450 selected GT nuclei matched the source proposals;17 were missed. Mean displacement2.3448 pixels, median1.8980, p90=4.3117 and p95=5.8393. The detector-held comparison trains separate GT-centered and Stage 1-centered models and evaluates both coordinate inputs at every epoch. Stage 1-centered training improves the Stage 1-evaluated mean by.005635, but interval[−.006667,.017857] crosses zero and apoptosis recall falls.166667. It fails promotion.

**Scope:** These classifier tests contain matched nuclei only. The full proposal manifest retains unmatched predictions and unmatched GT provenance; no conditional score is relabeled end-to-end. Full-proposal classification remains unexecuted. Fold0 selection was fixed from cohort size before classification results. No Stage 1 weights or decoder settings were changed.

### D. Is class balancing insufficient because ROIs dominate exposure?

**Reviewer:** A class-balanced sampler may still oversample a few large or distinctive ROIs. Correcting both priors and sampling can double-count imbalance.

**Architect:** Replace only the sampling distribution with uniform nonempty ROI-class buckets followed by uniform nuclei. Weight each row by inverse bucket size, draw the same number of samples, keep ordinary CE and record exact ROI, class, bucket and UID exposure.

**Evidence and decision:** Mean ROI F1=.084035, delta=−.000063, interval[−.005564,.005508], only one positive seed. Reject this sampler; no A3-plus-sampler combination is justified. The negative result does not erase exposure diagnostics or imply ROI grouping is unnecessary for evaluation.

### E. Does restricted encoder adaptation fix the representation?

**Reviewer:** Frozen-head gains do not establish the benefit of LoRA. Cache only an exactly equivalent frozen prefix and prove gradient flow in the adapted suffix.

**Architect:** Cache blocks0–19 outputs in FP32. Keep blocks20–23 and final norm; inject rank8, alpha16 Q/V-only LoRA. K, MLP, norms and other encoder weights remain frozen. The suffix plus head has224,474 trainable parameters:196,608 adapters and27,866 head. Use head LR.001, adapter LR1e-5, AdamW weight decay.01, effective batch64, microbatch4 with correct final-batch normalization, clip1. Three seeds, five epochs, same300/150 and matched five-epoch frozen controls. Encoder eval mode disables stochastic layers while retaining autograd.

**Evidence and decision:** Initial full-forward parity error0. All three final ROI F1 deltas are0; paired ROI interval[0,0]. Adapter gradients and norms are recorded, so the conclusion is lack of measurable finite-budget benefit, not proof of absent gradients. No gain justifies the conditional second configuration. No augmentation-enabled or longer LoRA claim is made.

### F. Does a clean predicted mask improve pooling?

**Reviewer:** Historical BioMask has indirect contamination through its proposal-generation dependencies. Good mask Dice cannot repair that lineage.

**Architect:** Train a new random-initialized mask-only network on 294 non-fold0 GT-centered crops and GT prompts. Exclude all 156 fold0 nuclei from every training dependency. No semantic class or Stage 1 proposal enters mask training. Freeze auxiliary heads, use.6 valid-pixel BCE+.4 soft Dice, AdamW3e-4, weight decay1e-4, batch16,10 fixed epochs, seed17. This is a new small-data control, not reproduction of the old mask recipe. Align predicted masks by integer crop-origin translation, average6×6 pixels per token, normalize weight mass, and record Gaussian fallback for empty mass. Compare A3, Gaussian, mask-only pooling, CLS-plus-mask mean and corresponding shuffled-mask controls on identical80/76 fold0 membership.

**Evidence:** Mask training objective falls from.521933 to.273805 over ten epochs. Held mean Dice=.783957, IoU=.669140. Mask-only pooling mean gain over A3=.000730, interval[−.030367,.028509], and it loses.011136 to Gaussian. Histiocyte and neutrophil recall fall.291667 and.194444. CLS-plus-mask loses.013995 to A3. Neither passes. Predicted masks are excluded from the final architecture; the clean provenance and negative evidence remain archived.

### Final confirmation: the candidate does not survive the guardrail

**Reviewer:** The internal gain is substantial, but a pooled average can hide class harm. Apply the prespecified confirmation rule without relaxing it.

**Architect:** Run A3 at fixed epoch 10 on300 training/150 reused development nuclei for all three seeds, comparing with preserved Exploration 3 A5 epoch10 predictions.

**Evidence:** ROI gains are[.012702839,.010800061,.024618805], mean.016040568. The descriptive interval is[−.00860277,.03925649]. Mean class recall changes in canonical order are[−.136752,+.088889,+.111111,0,+.222222,−.111111,−.027778,−.333333,+.027778,+.361111]. Tumor, neutrophil and epithelium exceed the allowed decline. Therefore A3 fails confirmation. The result does not authorize another representation, a new loss, a different epoch, or a class-specific threshold search on the same150 rows. The single next full-scale architecture is the retained Exploration 3 A5.

## Final architecture and loss

Let I be an ROI, c=(x,y) an immutable nucleus coordinate and C96(I,c) the96-pixel white-padded crop, resized bicubically to224² and normalized by ImageNet RGB mean/std. Let fθ be the verified frozen UNI2-h encoder. Define h=LN(fθ(C96)), with nonaffine1536-dimensional normalization. Let b contain the exact16 TierA RGB/point features from the saved schema, standardized using training-only mean and standard deviation floored at1e-6. The selected logits are

`z = W h + a + R ((P_h h) ⊙ (P_b b))`.

W has1536×10 weights and10 biases; P_h is1536→8, P_b16→8, R8→10, all biasless. R starts at zero. Total trainable parameters27,866. There is no mask branch, tissue branch, learned gate, multiscale branch, auxiliary target or LoRA in the recommendation. TierA is retained because this exploration held the previously selected head fixed; the diagnostic suggests its incremental value on a new representation is uncertain. This is a conservative continuity decision, not a proof that every retained feature is biologically causal.

For training counts n_c and N=Σn_c, sample nucleus i with probability proportional to1/n_yi. The induced class prior is uniform when all classes are present. Optimize ordinary cross-entropy `L=-z_y+log Σ_c exp(z_c)`. Do not additionally add log training priors or class weights. The resulting training distribution targets balanced classification and is not a calibrated estimate of the natural class prevalence. Evaluate NLL and15-bin ECE on the declared target population. Fit a scalar temperature only on a disjoint calibration split, preserve coordinates, and verify any confidence-order effect under the score-prioritized V17 matcher.

The prospective full-data recipe inherits AdamW LR.001, betas(.9,.999), eps1e-8, weight decay.01, batch64, clip1, at most20 epochs and patience5 with minimum improvement1e-4 on prespecified validation fixed-ten ROI Macro-F1. The finite probes used fixed epochs and no early stopping. The full-scale recipe is a blueprint, not an executed result. Frozen deterministic caches have no stain augmentation; adding stain augmentation would change the experimental contract and requires a new controlled study. Physical pixel size and patient grouping must be verified before external generalization claims.

## Gradient starvation and Occam's razor: what the evidence supports

Unbalanced CE minimizes an empirical risk dominated by frequent classes. It can yield low tail recall despite nonzero tail gradients. Adding handcrafted branches does not mathematically imply gradient starvation: branch scaling, normalization, initialization, loss distribution and label support determine gradient magnitudes. The old plateau therefore cannot be attributed causally to “biology fusion” without controls. The P1 logs show class exposure and positive-target gradients; the P2/P3 controls distinguish useful RGB corrections from placebo or shuffled inputs. Exploration 4 local-token gains suggest representation alignment mattered more than extra head capacity in this cohort, but failed confirmation prevents a universal conclusion.

UNI2 feature geometry is evaluated through class centroid scatter, same/different-class cosine similarity across ROIs and held-fold5NN purity. These are descriptive geometry diagnostics, not evidence that latent axes encode specific biology. LoRA changes Q/V subspaces with a low-rank update; class imbalance still enters through sampling and loss. A low-rank parameterization does not guarantee minority separability, and a five-epoch null result does not prove no adaptation can work. Every rejected component remains a recorded finite test rather than a broad theoretical prohibition.

## Literature grounding and source boundaries

The official [UNI repository](https://github.com/mahmoodlab/UNI) supplies the UNI2-h architecture and usage contract; local checkpoint shape and token extraction verify the actual implementation used here. [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) motivates separating register tokens from spatial patch tokens; it does not establish that any pooling rule wins on PUMA. [LoRA](https://arxiv.org/abs/2106.09685) defines low-rank adaptation; its language-model findings are not a pathology performance guarantee. Earlier primary-source notes on HoVer-Net/HoVer-NeXt, CoNIC, PanNuke, NuCLS, CONCH, imbalance losses and logit adjustment remain in the preserved Exploration 1–3 chapters and literature artifacts. No paper's reported benchmark score is substituted for a local controlled result.

## Implementation verification and limitations

Each material architecture has local source copies and entrypoints, exact configs, result indexes, checkpoint/prediction copies and provenance. Shared caches are referenced once by manifest and hash; code does not import another architecture's mutable modules. Frozen P4 package checkpoints are strictly loaded and their forward logits compared to archived predictions. The A3 raw-image candidate implementation reproduced cached pooling and two full-image UNI2 forwards with maximum error0, but is not the recommended model after confirmation failure. The final package retains the previously verified A5 implementation and complete-proposal evaluator.

Windows long-path failures, an initial duplicate NPZ keyword before A training, a harmless diagnostic console variable shadowing and the proposal hash-convention investigation are documented implementation incidents. No failed partial run was silently counted as a completed valid experiment. Missing historical executable versions and missing original metrics are explicitly labeled unresolved or NOT RECORDED IN ORIGINAL RUN. The archive preserves original reports and uncertain artifacts; it does not manufacture lineage.

The study's main limitations are enriched small-sample class distributions, reused development selection, unknown patient identifiers, sparse class support per ROI, exploratory multiple comparisons, GT-centered controls, conditional matched-proposal evaluation, no full-data adaptation trial and no external validation. These limitations bound the conclusions. The completed finite study selects one next experiment; it does not establish Q1 acceptance, clinical readiness or SOTA superiority.

## Exact endpoint tables

### A/B internal CV

Source: [seed_summary.csv](../../RESULTS/METRICS/A_complete/seed_summary.csv).

| family | ROI_F1_mean | ROI_F1_sd | delta_vs_A5 | positive_seeds | bootstrap_low | bootstrap_high | control_advantage | parameters | promoted |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.08409771473601262 | 0.0032421625811143674 | 0.0 | 0 | 0.0 | 0.0 |  | 27866 | False |
| A1 | 0.10173139705054597 | 0.0019961454060512304 | 0.017633682314533366 | 3 | 0.006668270854441071 | 0.02852200833051897 | 0.04848474614432059 | 27866 | True |
| A1_broken | 0.05324665090622538 | 0.000564082905222164 | -0.030851063829787233 | 0 | -0.04245862884160756 | -0.018636327817178888 |  | 27866 | False |
| A2 | 0.1066542834627941 | 0.0018771678834211662 | 0.022556568726781485 | 3 | 0.012306822019587979 | 0.03332443994146122 | 0.03746594618935045 | 27866 | True |
| A2_broken | 0.06918833727344365 | 0.0006625011680688243 | -0.014909377462568965 | 0 | -0.0269269109535067 | -0.0023319542947202593 |  | 27866 | False |
| A3 | 0.10875830237532368 | 0.00404174524471271 | 0.02466058763931105 | 3 | 0.013820612405718788 | 0.03609149499043116 | 0.05020038275357426 | 27866 | True |
| A3_broken | 0.058557919621749416 | 0.002858761157433871 | -0.0255397951142632 | 0 | -0.03710815602836879 | -0.013585303388494879 |  | 27866 | False |
| A4 | 0.10468647979286277 | 0.0030843226669976962 | 0.02058876505685016 | 3 | 0.011812197455814482 | 0.029449622875154798 | 0.018268602949454024 | 27866 | True |
| A4_broken | 0.08641787684340875 | 0.005137303584695099 | 0.002320162107396148 | 2 | -0.0069868569177079795 | 0.01178987954519868 |  | 27866 | False |
| B1 | 0.08433412135539797 | 0.002967664311686452 | 0.00023640661938535237 | 2 | -0.0013396375098502758 | 0.0017336485421591806 | -5.516154452323008e-05 | 40234 | False |
| B2 | 0.08572892040977148 | 0.0036822813218344223 | 0.0016312056737588648 | 3 | -2.3640661938534358e-05 | 0.003420015760441292 | 0.0013396375098502777 | 40234 | False |
| B_duplicate | 0.0843892828999212 | 0.0031453039677459904 | 0.0002915681639085917 | 2 | -0.0007328605200945628 | 0.0013711583924349879 |  | 40234 | False |

### ROI-class sampler and other completed CV controls

Source: [seed_summary.csv](../../RESULTS/METRICS/D_complete/seed_summary.csv).

| family | ROI_F1_mean | ROI_F1_sd | delta_vs_A5 | positive_seeds | bootstrap_low | bootstrap_high | control_advantage | parameters | promoted |
|---|---|---|---|---|---|---|---|---|---|
| A0 | 0.08409771473601262 | 0.0032421625811143674 | 0.0 | 0 | 0.0 | 0.0 |  | 27866 | False |
| A1 | 0.10173139705054597 | 0.0019961454060512304 | 0.017633682314533366 | 3 | 0.006668270854441071 | 0.02852200833051897 | 0.04848474614432059 | 27866 | True |
| A1_broken | 0.05324665090622538 | 0.000564082905222164 | -0.030851063829787233 | 0 | -0.04245862884160756 | -0.018636327817178888 |  | 27866 | False |
| A2 | 0.1066542834627941 | 0.0018771678834211662 | 0.022556568726781485 | 3 | 0.012306822019587979 | 0.03332443994146122 | 0.03746594618935045 | 27866 | True |
| A2_broken | 0.06918833727344365 | 0.0006625011680688243 | -0.014909377462568965 | 0 | -0.0269269109535067 | -0.0023319542947202593 |  | 27866 | False |
| A3 | 0.10875830237532368 | 0.00404174524471271 | 0.02466058763931105 | 3 | 0.013820612405718788 | 0.03609149499043116 | 0.05020038275357426 | 27866 | True |
| A3_broken | 0.058557919621749416 | 0.002858761157433871 | -0.0255397951142632 | 0 | -0.03710815602836879 | -0.013585303388494879 |  | 27866 | False |
| A4 | 0.10468647979286277 | 0.0030843226669976962 | 0.02058876505685016 | 3 | 0.011812197455814482 | 0.029449622875154798 | 0.018268602949454024 | 27866 | True |
| A4_broken | 0.08641787684340875 | 0.005137303584695099 | 0.002320162107396148 | 2 | -0.0069868569177079795 | 0.01178987954519868 |  | 27866 | False |
| B1 | 0.08433412135539797 | 0.002967664311686452 | 0.00023640661938535237 | 2 | -0.0013396375098502758 | 0.0017336485421591806 | -5.516154452323008e-05 | 40234 | False |
| B2 | 0.08572892040977148 | 0.0036822813218344223 | 0.0016312056737588648 | 3 | -2.3640661938534358e-05 | 0.003420015760441292 | 0.0013396375098502777 | 40234 | False |
| B_duplicate | 0.0843892828999212 | 0.0031453039677459904 | 0.0002915681639085917 | 2 | -0.0007328605200945628 | 0.0013711583924349879 |  | 40234 | False |
| D1 | 0.0840346729708432 | 0.0067042155233469165 | -6.304176516941835e-05 | 1 | -0.005563632781717889 | 0.005508471237194638 |  | 27866 | False |

### Plain-head representation diagnostic

Source: [plain_head_endpoints.csv](../../RESULTS/METRICS/TARGET_POOLING/DIAGNOSTICS/plain_head_endpoints.csv).

| representation | seed | ROI_F1 | semantic_F1 | NLL | ECE15 | parameters |
|---|---|---|---|---|---|---|
| CLS | 17 | 0.07510638297872341 | 0.3828255229493534 | 1.707814335823059 | 0.08728231141964593 | 15370 |
| CLS | 29 | 0.07491725768321514 | 0.3994125047129147 | 1.7472174167633057 | 0.12000165566802025 | 15370 |
| CLS | 43 | 0.07560283687943263 | 0.3987526684562966 | 1.7416030168533325 | 0.08655598700046539 | 15370 |
| A3 | 17 | 0.10940222897669707 | 0.5578480097588405 | 1.3856940269470215 | 0.11609196742375692 | 15370 |
| A3 | 29 | 0.11164809186085782 | 0.5829810550744201 | 1.3491835594177246 | 0.11315871899326643 | 15370 |
| A3 | 43 | 0.10293144208037826 | 0.5328759680416407 | 1.4028531312942505 | 0.14325126757224402 | 15370 |

### Proposal alignment

Source: [alignment_endpoints.csv](../../RESULTS/METRICS/STAGE1_PROPOSALS/alignment_endpoints.csv).

| training | seed | eval_coordinates | ROI_F1 | semantic_F1 | accuracy | NLL | ECE15 | scope |
|---|---|---|---|---|---|---|---|---|
| C_GTTRAIN | 17 | GT_centered | 0.23900226757369616 | 0.5922029760520809 | 0.64 | 1.4226897954940796 | 0.19050967613855999 | matched conditional detector-heldfold0 |
| C_GTTRAIN | 17 | Stage1_centered | 0.21826330532212884 | 0.5475260921232789 | 0.6 | 1.4515026807785034 | 0.146475949883461 | matched conditional detector-heldfold0 |
| C_GTTRAIN | 29 | GT_centered | 0.21753968253968256 | 0.5277805667508184 | 0.5733333333333334 | 1.4021797180175781 | 0.10924008965492248 | matched conditional detector-heldfold0 |
| C_GTTRAIN | 29 | Stage1_centered | 0.19968253968253968 | 0.4855629735425132 | 0.5466666666666666 | 1.3764334917068481 | 0.14792802393436433 | matched conditional detector-heldfold0 |
| C_GTTRAIN | 43 | GT_centered | 0.2255422168867547 | 0.5552105011991941 | 0.6 | 1.3996933698654175 | 0.12077985485394796 | matched conditional detector-heldfold0 |
| C_GTTRAIN | 43 | Stage1_centered | 0.2083503401360544 | 0.5068410186304924 | 0.5733333333333334 | 1.3794034719467163 | 0.15352481762568157 | matched conditional detector-heldfold0 |
| C_STAGE1TRAIN | 17 | GT_centered | 0.24471655328798186 | 0.6045699228307925 | 0.6533333333333333 | 1.396969199180603 | 0.16050316214561464 | matched conditional detector-heldfold0 |
| C_STAGE1TRAIN | 17 | Stage1_centered | 0.20326330532212883 | 0.5164249639249638 | 0.5733333333333334 | 1.4455325603485107 | 0.1489503530661265 | matched conditional detector-heldfold0 |
| C_STAGE1TRAIN | 29 | GT_centered | 0.21039682539682542 | 0.5181203337725078 | 0.5733333333333334 | 1.3711615800857544 | 0.11934932192166646 | matched conditional detector-heldfold0 |
| C_STAGE1TRAIN | 29 | Stage1_centered | 0.22063492063492066 | 0.5247339693106283 | 0.5866666666666667 | 1.3691726922988892 | 0.15369859258333843 | matched conditional detector-heldfold0 |
| C_STAGE1TRAIN | 43 | GT_centered | 0.23554221688675475 | 0.5902064765300059 | 0.6266666666666667 | 1.3929980993270874 | 0.1696432109673818 | matched conditional detector-heldfold0 |
| C_STAGE1TRAIN | 43 | Stage1_centered | 0.21930272108843543 | 0.5500074332272474 | 0.5866666666666667 | 1.3928828239440918 | 0.13248253206412 | matched conditional detector-heldfold0 |

### Restricted LoRA

Source: [endpoints.csv](../../RESULTS/METRICS/LORA/endpoints.csv).

| seed | family | epoch | ROI_F1 | semantic_F1 | NLL | ECE15 | gap |
|---|---|---|---|---|---|---|---|
| 17 | LoRA | 5 | 0.14064529914529916 | 0.5544376598171665 | 1.3225759267807007 | 0.12434486885865528 | 0.2618303214383957 |
| 17 | frozen | 5 | 0.14064529914529916 | 0.5544376598171665 | 1.322662115097046 | 0.12433875679969789 | 0.2618303214383957 |
| 29 | LoRA | 5 | 0.15778174603174605 | 0.5984242809687454 | 1.223506212234497 | 0.06919079244136812 | 0.20904586987425755 |
| 29 | frozen | 5 | 0.15778174603174605 | 0.5984242809687454 | 1.2235568761825562 | 0.0691632095972697 | 0.20904586987425755 |
| 43 | LoRA | 5 | 0.15938888888888886 | 0.6271098174566112 | 1.2631118297576904 | 0.09717006246248881 | 0.19804921025703837 |
| 43 | frozen | 5 | 0.15938888888888886 | 0.6271098174566112 | 1.2631922960281372 | 0.09717514038085938 | 0.19804921025703837 |

### Clean mask pooling

Source: [endpoints.csv](../../RESULTS/METRICS/BIOMASK_GUIDED/endpoints.csv).

| family | seed | ROI_F1 | semantic_F1 | NLL | ECE15 | gap |
|---|---|---|---|---|---|---|
| F_A3 | 17 | 0.22297619047619044 | 0.5355938505938506 | 1.5013275146484375 | 0.13103641236298963 | 0.3521960044786132 |
| F_A3 | 29 | 0.22111111111111112 | 0.5484337884337884 | 1.3056923151016235 | 0.08902277640606228 | 0.3889282062966274 |
| F_A3 | 43 | 0.23833333333333337 | 0.5800082206795761 | 1.3965457677841187 | 0.11855124348872588 | 0.36228402955385053 |
| F_GAUSSIAN | 17 | 0.2517857142857143 | 0.5966662980981867 | 1.4412832260131836 | 0.1436850679548163 | 0.2548857772774181 |
| F_GAUSSIAN | 29 | 0.23253968253968252 | 0.5861260568300547 | 1.2298712730407715 | 0.1497513945949705 | 0.3421274635287287 |
| F_GAUSSIAN | 43 | 0.2336904761904762 | 0.5718018575851394 | 1.3252644538879395 | 0.08037904767613661 | 0.37958563074352547 |
| F_MASK | 17 | 0.2116300366300366 | 0.5041182099634112 | 1.4344053268432617 | 0.14960558124278717 | 0.38102814522938355 |
| F_MASK | 29 | 0.23849206349206353 | 0.5955974749887794 | 1.220354437828064 | 0.11371034266133058 | 0.33451501972897857 |
| F_MASK | 43 | 0.23448717948717954 | 0.5527253938012453 | 1.295032262802124 | 0.15535379436455274 | 0.3946366009291705 |
| F_CLS_MASK | 17 | 0.20777777777777778 | 0.5459246465384574 | 1.4210395812988281 | 0.13227741635943716 | 0.3775169119031011 |
| F_CLS_MASK | 29 | 0.20778911564625852 | 0.5439446124248459 | 1.2601425647735596 | 0.1311302153687728 | 0.3857274801884232 |
| F_CLS_MASK | 43 | 0.2248701298701299 | 0.5276093885830193 | 1.351241111755371 | 0.10819450314891967 | 0.4098918002865908 |
| F_SHUFFLE | 17 | 0.20974025974025973 | 0.4835518207282913 | 1.4699019193649292 | 0.20342382710230977 | 0.4372426788897379 |
| F_SHUFFLE | 29 | 0.22706349206349205 | 0.5793803418803419 | 1.2725512981414795 | 0.15197186152401723 | 0.3723298971983181 |
| F_SHUFFLE | 43 | 0.20891774891774895 | 0.48324115971174786 | 1.384325623512268 | 0.12881655834223094 | 0.4195114079819964 |
| F_CLS_SHUFFLE | 17 | 0.1925830332132853 | 0.5019829877724614 | 1.4386078119277954 | 0.18318885094241094 | 0.44672821330716084 |
| F_CLS_SHUFFLE | 29 | 0.20921150278293138 | 0.5161760461760462 | 1.2979379892349243 | 0.1197843024213063 | 0.4358952262048238 |
| F_CLS_SHUFFLE | 43 | 0.23558441558441562 | 0.5334180278281911 | 1.4049774408340454 | 0.11378204626472371 | 0.41410822369806044 |

### Final confirmation

Source: [confirmation_endpoints.csv](../../RESULTS/METRICS/confirmation_endpoints.csv).

| family | seed | epoch | ROI_F1 | semantic_F1 | NLL | ECE15 | gap |
|---|---|---|---|---|---|---|---|
| P4_A3 | 17 | 10 | 0.15821474358974358 | 0.5685924486956069 | 1.240935206413269 | 0.11130913237730662 | 0.4011505168658205 |
| P3_A5 | 17 | 10 | 0.1455119047619048 | 0.565542450844175 | 1.2713536024093628 | 0.08611232390006383 | 0.3662069550416608 |
| P4_A3 | 29 | 10 | 0.15595482295482296 | 0.5854633196222402 | 1.241654872894287 | 0.09986207952102025 | 0.3745695005233619 |
| P3_A5 | 29 | 10 | 0.1451547619047619 | 0.5741634753740466 | 1.3134123086929321 | 0.08755476872126261 | 0.33174809187768817 |
| P4_A3 | 43 | 10 | 0.16514856711915538 | 0.6225763097638924 | 1.2364873886108398 | 0.1202930321296056 | 0.3303369315437765 |
| P3_A5 | 43 | 10 | 0.1405297619047619 | 0.5761960403817878 | 1.2722482681274414 | 0.1124365015824636 | 0.3468909448097076 |


## Architecture package directory

- [A0](../ARCHITECTURES/A0/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [A1](../ARCHITECTURES/A1/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [A2](../ARCHITECTURES/A2/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [A3](../ARCHITECTURES/A3/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [A4](../ARCHITECTURES/A4/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [B1](../ARCHITECTURES/B1/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [B2](../ARCHITECTURES/B2/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [B_duplicate](../ARCHITECTURES/B_duplicate/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [C_ALIGNMENT](../ARCHITECTURES/C_ALIGNMENT/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [D1](../ARCHITECTURES/D1/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [DIAG_A3LINEAR](../ARCHITECTURES/DIAG_A3LINEAR/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [E0_FROZEN](../ARCHITECTURES/E0_FROZEN/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [E1_LORA](../ARCHITECTURES/E1_LORA/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [F_A3](../ARCHITECTURES/F_A3/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [F_CLEAN_MASK_NETWORK](../ARCHITECTURES/F_CLEAN_MASK_NETWORK/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [F_CLS_MASK](../ARCHITECTURES/F_CLS_MASK/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [F_CLS_SHUFFLE](../ARCHITECTURES/F_CLS_SHUFFLE/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [F_GAUSSIAN](../ARCHITECTURES/F_GAUSSIAN/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [F_MASK](../ARCHITECTURES/F_MASK/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [F_SHUFFLE](../ARCHITECTURES/F_SHUFFLE/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [FINAL_A3](../ARCHITECTURES/FINAL_A3/ARCHITECTURE.md) — local code, configuration, results and provenance.
- [FINAL_RECOMMENDED_A5](../ARCHITECTURES/FINAL_RECOMMENDED_A5/ARCHITECTURE.md) — local code, configuration, results and provenance.
