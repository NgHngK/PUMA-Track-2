# Exploration 6: PUMA Stage-2 study — final report

## Executive conclusion

I completed the amended Exploration 6 model-selection sequence and evaluated both frozen holdouts on an NVIDIA L4. The selected Stage-2 classifier uses a **frozen UNI2-h encoder, Gaussian token pooling (sigma 1.5), the rank-8 appearance × Tier-A interaction head, inverse-frequency sampling, ordinary cross-entropy, FOV 96, and a fixed 10-epoch schedule**. Seed 17 is the checkpoint chosen for future deployment testing, and I did not use an ensemble. Its SHA256 is `269e5e01face1c5d65a61228f0dbcca7d7a8afed8ddaf773c9f16119926edd63`.

Gaussian pooling produced a stable improvement over the frozen CLS baseline on the D900 development design: mean paired ROI-F1 gain **+0.028923**, 95% paired ROI bootstrap interval **[+0.014916, +0.043810]**, with positive effects in all three seeds, no mean class-recall loss above 0.10, and mean generalization-gap reduction **−0.106719**. The one-shot confirmation ROI fixed-10 macro-F1 was **0.183281**.

The locked natural population gives a more cautious result. ROI fixed-10 macro-F1 fell to **0.157786**, and semantic accuracy was **0.531180**. The inverse-sampled classifier overpredicted several rare classes and recovered only **51.0%** of the dominant tumor class, although tumor precision was **98.9%**. Neutrophil and epithelium had no ground-truth examples in this holdout, so their recall cannot be estimated. The model still predicted 47 neutrophils and 517 epithelium examples, all counted as false positives in this annotation population. Exploration 6 therefore supports Gaussian local pooling for this conditional-classification design, but it does not establish population calibration, end-to-end Stage-1+Stage-2 performance, patient-level generalization, clinical readiness, or SOTA performance.

## Study integrity and artifact lineage

The experiment ledger contains **103 completed training runs** and **1,395 joint-training epochs**. The 12 Stage-C runs additionally contain **30 classifier-retraining epochs**, yielding **1,425 fitted epochs** when the decoupled phases are counted. Verified equivalent completed runs were reused; interrupted directories were not automatically restarted. Original Explorations 1–5 artifacts remained unchanged.

The final recipe was frozen before either holdout at `FINAL_RECIPE_FREEZE.json`. The freeze names `p6_F_gaussian_s17`, config SHA256 `3b3ba817a8dbe98265b428f73ea59cd93605145312f5ef0dff35af07412820e3`, and the checkpoint hash above. The confirmation opening marker was written after the freeze, followed by the locked-natural opening marker. `I_FINAL_HOLDOUT_RESULTS.json` records `post_holdout_model_changes=false`. Both holdouts were evaluated once.

The execution environment was NVIDIA L4, PyTorch 2.11.0+cu128, and Python 3.13.15. Raw-versus-cached prediction parity produced zero DEV class disagreements for all three seeds; maximum absolute logit error was 2.38e-6, 2.86e-6, and 4.05e-4 for seeds 17, 29, and 43.

## A. Data scaling

Mean D300/D600/D900 development ROI fixed-10 F1 was **0.129493 / 0.133084 / 0.149770**. D900 minus D300 was **+0.020278**, with ROI bootstrap interval **[+0.007568, +0.033165]** and three positive seeds. Neutrophil recall decreased by **0.240741**, however, so the scaling comparison failed the predeclared class-harm guard. D900 remained the fixed study population because it was the preregistered complete training scale for the subsequent controls; the result is mixed evidence rather than a clean claim that more rows solved rare-class learning.

## B. Raw augmentation and cache parity

| Arm | Mean DEV ROI-F1 | Mean semantic F1 | Mean gap | Decision |
|---|---:|---:|---:|---|
| AUG0: no augmentation | 0.149770 | 0.427800 | 0.478312 | Retained |
| AUG1: geometry | 0.144632 | 0.411262 | 0.438218 | Rejected |
| AUG2: geometry + stain | 0.153104 | 0.430919 | 0.418438 | Rejected versus AUG0 |

AUG1 minus AUG0 was **−0.005139**, with interval **[−0.012938, +0.003296]**, one positive seed, and melanophage recall harm. AUG2 minus AUG0 reached the minimum point gain (**+0.003333**) and two positive seeds with no class harm, but its interval **[−0.005616, +0.012262]** crossed zero. AUG2 beat AUG1, but AUG1 was not the retained control. The frozen outcome was therefore AUG0.

## C. Long-tail sampling and classifier retraining

The retained inverse-frequency sampler with ordinary CE achieved mean ROI-F1 **0.149770**. All alternatives were lower: natural CE **0.138005**, tempered sampling **0.140028**, cRT with reset **0.137769**, and cRT without reset **0.140762**. cRT reset reduced the mean generalization gap from about 0.478 to 0.390 but sacrificed the primary ROI metric, so gap reduction alone did not justify promotion. Inverse sampling plus ordinary CE remained frozen.

## D. Optimizer and regularization study

The three-fold control mean ROI-F1 was **0.241378**. Optimizer trial 006 was the best eligible optimizer candidate at **0.249138** (gain **+0.007760**, SD 0.022992, no class-harm flag), using learning rate 0.00118379, weight decay 0.000110388, batch 32, cosine scheduling, max 20 epochs, and patience 5. Trial 007 scored higher at 0.252959 but was ineligible because of apoptosis recall harm.

None of the nine regularization trials beat optimizer trial 006; the highest scored 0.240783 and also harmed plasma-cell recall. The single external-DEV finalist from optimizer trial 006 scored **0.151088** versus baseline **0.148937**, a paired gain of only **+0.002151** with interval **[−0.016173, +0.021845]**. It harmed lymphocyte, stroma, and apoptosis recall. Stage D therefore rejected the HPO finalist and returned to the original fixed 10-epoch baseline recipe.

## E. Tier-A audit

Removing Tier-A reduced seed-17 ROI-F1 from **0.148937 to 0.136437**: paired change **−0.012500**, interval **[−0.030001, +0.006964]**. Lymphocyte recall fell by 0.521739 and stroma recall by 0.111111. Tier-A was retained.

The association audit also shows why Tier-A should not be interpreted as purely causal biology. TRAIN ROI eta-squared reached **0.600932** for H_p90, **0.596443** for H_mean, and **0.539665** for H_p50. H_std had the strongest class association (eta-squared **0.250131**). The largest absolute DEV correctness correlation was modest (H_entropy32, **−0.143884**). Patient, case, slide, stain-batch, and acquisition identifiers were unavailable, so ROI associations cannot be separated cleanly into biology and nuisance structure.

## F. Frozen representation comparison

| Representation | Seed-17 ROI-F1 | Gain over CLS | 95% paired ROI interval | Class harm | Provisional gate |
|---|---:|---:|---:|---|---|
| CLS control | 0.148937 | — | — | — | Reference |
| Gaussian, sigma 1.5 | 0.181071 | +0.032134 | [+0.003094, +0.060049] | None | Pass |
| Neighborhood, radius 1 | 0.172798 | +0.023861 | [−0.003929, +0.049566] | None | Fail interval |
| Fixed global-local | 0.178095 | +0.029158 | [+0.006478, +0.052435] | None | Pass |

Gaussian was selected because it had the highest eligible ROI-F1. The scalar global-local mixture remained excluded by the preregistered finite grid.

## G. Conditional encoder adaptation

The conditional LoRA gate closed. The selected frozen representation still had DEV ROI-F1 **0.181071**, while its semantic generalization gap was **0.362721**. That gap left limited diversity and memorization as reasonable explanations for the remaining failure. The amended gate protocol required the contrary finding before another encoder-adaptation experiment could run. Consequently, zero new LoRA candidates were fit; this is the correct terminal outcome of the conditional stage, not a missing experiment. The previously failed Exploration-4 LoRA configuration was not repeated.

## H. Three-seed replication and final recipe

| Seed | Gaussian ROI-F1 | CLS ROI-F1 | Paired gain |
|---:|---:|---:|---:|
| 17 | 0.181071 | 0.148937 | +0.032134 |
| 29 | 0.177202 | 0.146616 | +0.030587 |
| 43 | 0.177806 | 0.153759 | +0.024048 |

The mean gain was **+0.028923**, with interval **[+0.014916, +0.043810]** and three positive seeds. Mean class-recall changes were −0.0172 tumor, −0.0435 lymphocyte, +0.1481 plasma cell, +0.0185 histiocyte, +0.0926 melanophage, +0.3148 neutrophil, +0.2593 stroma, +0.0556 epithelium, +0.1667 endothelium, and +0.0926 apoptosis. No class crossed the −0.10 harm threshold. The promotion gate passed, and the seed-17 Gaussian checkpoint was frozen prospectively without ensembling.

Seed 17 was also used in Stage-F selection, while seeds 29 and 43 were newly fit confirmations. This predeclared design is stronger than a single-seed result, but the aggregate interval is not equivalent to three wholly independent external replications.

## I. Frozen holdouts

| Metric | Confirmation (n=225) | Locked natural (n=10,712) |
|---|---:|---:|
| ROI fixed-10 macro-F1 | 0.183281 | 0.157786 |
| Semantic macro-F1, fixed 10 | 0.573054 | 0.209392 |
| Semantic macro-F1, supported classes | 0.573054 | 0.261740 |
| Balanced accuracy, fixed 10 | 0.579379 | 0.457669 |
| Balanced accuracy, supported classes | 0.579379 | 0.572087 |
| Accuracy | 0.586667 | 0.531180 |
| NLL | 1.385635 | 1.545238 |
| ECE15 | 0.156726 | 0.141573 |

The ROI score and semantic score answer different questions. Semantic macro-F1 classifies the labeled nuclei directly. The PUMA fixed-10 metric averages class-aware matching within ROIs over the fixed ten-class universe, so it is substantially lower and is the primary study metric.

### Class-level semantic results

| Class | Confirmation support | Confirmation F1 | Natural support | Natural predicted | Natural precision | Natural recall | Natural F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Tumor | 59 | 0.6916 | 8,817 | 4,550 | 0.9890 | 0.5104 | 0.6733 |
| Lymphocyte | 22 | 0.4681 | 1,177 | 1,540 | 0.5299 | 0.6933 | 0.6007 |
| Plasma cell | 18 | 0.6452 | 8 | 378 | 0.0106 | 0.5000 | 0.0207 |
| Histiocyte | 18 | 0.3902 | 231 | 1,436 | 0.0940 | 0.5844 | 0.1620 |
| Melanophage | 18 | 0.8718 | 11 | 200 | 0.0350 | 0.6364 | 0.0664 |
| Neutrophil | 18 | 0.7333 | 0 | 47 | Not estimable | Not estimable | Not estimable |
| Stroma | 18 | 0.4242 | 278 | 615 | 0.1837 | 0.4065 | 0.2531 |
| Epithelium | 18 | 0.4865 | 0 | 517 | Not estimable | Not estimable | Not estimable |
| Endothelium | 18 | 0.4706 | 129 | 501 | 0.1477 | 0.5736 | 0.2349 |
| Apoptosis | 18 | 0.5490 | 61 | 928 | 0.0442 | 0.6721 | 0.0829 |

The natural shift is not a blanket loss of representation. Tumor and lymphocyte F1 remain comparatively strong, and several rare classes retain moderate recall. The dominant error is **precision collapse from severe overprediction under the natural class prior**. For example, 8 plasma cells generated 378 plasma predictions, 11 melanophages generated 200 predictions, and 61 apoptotic nuclei generated 928 predictions. This pattern is consistent with a classifier trained under inverse-frequency replacement sampling being used without a separately admitted natural-prior calibration or boundary correction. It does not prove that resampling alone is causal, because the natural set also differs in ROI composition and lacks two classes.

The natural support-aware metrics exclude neutrophil and epithelium only because their recall cannot be estimated. The official fixed-10 PUMA score correctly retains the fixed class universe and counts their predictions as false positives where applicable. No post-holdout model, threshold, temperature, or selection change was made.

## J. Fixed Stage-1 proposal evaluation

Stage J was not executable. The uploaded package did not contain a certified complete fixed Stage-1 V17 proposal manifest/cache with proposal UIDs and evaluator ground truth. The study recorded this as a fixed-artifact infeasibility and did not synthesize proposals, alter Stage-1 settings, or substitute a different population. Consequently, the reported metrics remain Stage-2 conditional-classification results and cannot be treated as end-to-end detector performance.

## Final scientific interpretation

Exploration 6 gives three useful results. First, smooth local Gaussian pooling extracts more useful nucleus-centered information from the frozen UNI2 tokens than CLS alone in the D900 design, and the improvement appears across all prescribed seeds. Second, Tier-A still adds useful information, although its strong ROI structure prevents a causal biological interpretation. Third, the natural holdout shows that the inverse-sampled decision boundary is poorly matched to natural prevalence: it often preserves rare-class recall by accepting many false positives and lower tumor recall.

I retained the frozen Gaussian/Tier-A model because it passed every pre-holdout selection gate, and the protocol does not allow changing models after seeing the holdout. The natural-population result limits what this decision means: the model is a research checkpoint for a separate external evaluation, not a deployment-ready system.

Future work should first recover and certify the complete fixed Stage-1 proposal contract, obtain real patient/case/slide identifiers, and build a naturally distributed development/calibration cohort with support for all ten classes. With those inputs, test a single predeclared prior or calibration correction while keeping Gaussian pooling frozen, then evaluate the complete proposal population end to end. That work is outside the completed Exploration-6 grid and must not reuse either Exploration-6 holdout for selection.

## Deliverables

The authoritative results are in `D_FINAL_SELECTION.json`, `E_FINAL_SELECTION.json`, `F_FINAL_SELECTION.json`, `G_FINAL_SELECTION.json`, `H_FINAL_SELECTION.json`, `FINAL_RECIPE_FREEZE.json`, `I_FINAL_HOLDOUT_RESULTS.json`, and `J_FINAL_RESULT.json`. Reproduction artifacts include `PROMPT6_EXPERIMENT_LEDGER.csv`, `PROMPT6_EPOCH_LEDGER.jsonl`, the additive decoupled-phase ledger, `PROMPT6_CLASS_LEDGER.csv`, `FINAL_ARTIFACT_SHA256.json`, and `STANDALONE_ARCHITECTURE_PACKAGE`.
