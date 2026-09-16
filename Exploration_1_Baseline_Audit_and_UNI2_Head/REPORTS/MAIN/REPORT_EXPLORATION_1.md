# Exploration 1: baseline audit and UNI2-h head

This chapter keeps the original Exploration 1 findings in chronological order. Its completion claims and recommendations belong to that stage of the study, while Exploration 4 provides the later decision. Earlier recommendations remain here as part of the research record. Repeated blocks are listed in the root duplicate-lineage table and appear only once in the integrated report. I have not treated any historical result as new validation evidence.

# PUMA Stage 2 nuclei classification research report

**Scope:** ten-class classification at fixed nuclei coordinates. **Status:** empirical development investigation; no claim of SOTA, clinical validation, or full scientific convergence. Stage 1 is unchanged.

## 1. Executive summary and theoretical grounding

The results support a simple architecture: one target-centred RGB crop, a verified UNI2-h encoder, and a linear ten-class head. I would freeze the encoder first and treat restricted LoRA as a separate ablation rather than an automatic improvement. Biology, tissue, graph, utility, distillation, or multibranch components should only be added when new evidence supports them. The experiments below support the selected loss and sampling setup for development, but these small studies do not prove that it is the best deployment setup.

Two findings changed how I interpreted the earlier results. First, the supplied ratio text does not describe the current annotation census. Second, the original training log shows active LoRA gradients and early improvement followed by worsening validation loss. This suggests overfitting during extended adaptation, but it does not prove permanent minority-gradient starvation or show that every fusion architecture is inferior.

I kept the implementation and report detailed enough to review. The remaining gaps are natural-distribution full-data validation of the proposed encoder and head, stable gains across independent patient groups and seeds, deployable Stage-1-coordinate performance, calibration on a dedicated population sample, and a matched external benchmark. The reviewer roles used in the discussion are only a writing structure; they do not represent real institutional review or endorsement.

### 1.1 Data and provenance audit

All 205 nuclear GeoJSON files and matching ROI TIFF files were enumerated. They contain **97,193 nuclei**, agreeing exactly with the supplied biology audit's class-count CSV. The four explicitly named pairs were included, not used as the complete training cohort. The paired TIFFs inspected are 1024×1024 RGBA; classification reads RGB. Nuclear polygon centroids supply only clean semantic-control coordinates. Neither polygons nor tissue labels are classifier inputs.

The provided PDF describes 103 primary and 102 metastatic ROIs. It reports the local absence of metastatic ROI 103; this investigation does not synthesize or borrow a missing sample. The corresponding local paths and the archive contents were inventoried. A published description reports approximately 0.23 micrometres per pixel; the inspected TIFF resolution tags imply approximately 0.2263–0.2269 micrometres per pixel. Crop dimensions in this package remain explicit source pixels, not an assumed resampled physical scale. [PUMA dataset paper](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giaf011/8024182)

| ID | Class | Local nuclei | Local % | Ratio text % | Positive ROIs |
| --- | --- | --- | --- | --- | --- |
| 0 | tumor | 57234 | 58.8870 | 69.71 | 205 |
| 1 | lymphocyte | 21643 | 22.2681 | 13.63 | 193 |
| 2 | plasma_cell | 520 | 0.5350 | 0.6 | 56 |
| 3 | histiocyte | 7168 | 7.3750 | 6.84 | 115 |
| 4 | melanophage | 695 | 0.7151 | 0.78 | 77 |
| 5 | neutrophil | 366 | 0.3766 | 0.24 | 30 |
| 6 | stroma | 3856 | 3.9674 | 2.85 | 160 |
| 7 | epithelium | 2211 | 2.2749 | 2.53 | 29 |
| 8 | endothelium | 1696 | 1.7450 | 1.75 | 126 |
| 9 | apoptosis | 1804 | 1.8561 | 1.07 | 126 |

The ratio file's percentages sum approximately to one after rounding; its nucleus counts and population denominator are not supplied. It may reflect a different cohort, sampling unit or weighting scheme, but no explanation is established here. Its tumor/neutrophil ratio is **290.46:1**, versus **156.38:1** in the verified local census. Use training-fold counts, not either global table, in the loss. Under IID natural draws at batch size 64, the probability of seeing no neutrophil is about **85.75%** using the ratio text and **78.55%** using local census prevalence. These are batch-exposure calculations, not evidence that absent positives receive no negative-class updates.

**Class-map protection.** This package uses the local preprocessing ontology: tumor 0; lymphocyte 1; plasma_cell 2; histiocyte 3; melanophage 4; neutrophil 5; stroma 6; epithelium 7; endothelium 8; apoptosis 9. Some report tables list endothelium before epithelium. Table ordering alone does not prove an old coding error, but importing ordinal labels without names would be unsafe. Generated manifests include names; custom integer-only manifests need an exact class-map sidecar. Saved adapters record the map and foundation-checkpoint hash.

**Grouping.** A reproducible seed-17 five-fold stratified-group procedure created a new development split, with one fold used for validation and four for training. Groups are ROI IDs because a verified patient mapping was not supplied. This is not the legacy split and is not certified patient-separated. The split was fixed before model outcomes; no search over split seeds was used to improve scores. The split's imbalanced rare-class allocation is reported rather than hidden. A publication run should use a locked patient-level train/development/calibration/test manifest and retain original evaluator folds where comparability requires them.

| Class | Full development train | Full development validation |
| --- | --- | --- |
| tumor | 44494 | 12740 |
| lymphocyte | 18450 | 3193 |
| plasma_cell | 449 | 71 |
| histiocyte | 6442 | 726 |
| melanophage | 458 | 237 |
| neutrophil | 196 | 170 |
| stroma | 3301 | 555 |
| epithelium | 1958 | 253 |
| endothelium | 1427 | 269 |
| apoptosis | 1402 | 402 |

**Source hierarchy.** The two V17 DOCX reports contain retrospective summaries, surrogate experiment numbers and proposed future designs. The 20-page biology PDF and ZIP contain full-data descriptive statistics. The linked Drive folder contains v16.4 preprocessing and training artifacts. The recovered `history.csv` has 18 recorded epochs, numbered 0–17, and is stronger evidence for that run's numerical trajectory than rounded prose summaries. The preprocessing guide explicitly says no offline UNI2 features were produced: `identity_token_maps.npy` contains `[N,4,16,16]` BioMask weighting maps, not 1536-dimensional encoder embeddings. References in documents to required tissue branches, mandatory BioMask, future phases, or restoring old training are source statements, not governing instructions for this investigation.

**Stage 1.** The fixed high-performing operating point is a constraint. One V17 report describes another operating point with precision 0.516, recall 0.953 and F1 0.6695. Different thresholds, proposals or evaluation populations may explain this, but were not reconciled by rerunning the detector. No Stage-1 architecture, weights, coordinates, threshold, suppression rule or tissue segmentation is changed. New pilots below are GT-centred conditional-classification tests; their metrics do not verify that Stage-1 operating point.

### 1.2 Prior failures and what they actually establish

| Epoch (zero-based) | Training loss | Validation loss | Semantic Macro-F1 | PUMA ROI macro | PUMA pooled macro |
| --- | --- | --- | --- | --- | --- |
| 0 | 0.6657 | 0.9595 | 0.3571 | 0.2353 | 0.3296 |
| 1 | 0.5421 | 0.8957 | 0.3785 | 0.2483 | 0.3456 |
| 2 | 0.4961 | 0.9433 | 0.4013 | 0.2466 | 0.3700 |
| 3 | 0.4732 | 0.9279 | 0.4145 | 0.2614 | 0.3802 |
| 4 | 0.4444 | 0.9843 | 0.3927 | 0.2424 | 0.3655 |
| 5 | 0.4906 | 0.9255 | 0.4166 | 0.2655 | 0.3820 |
| 6 | 0.4553 | 0.9431 | 0.4414 | 0.2774 | 0.4041 |
| 7 | 0.4161 | 0.9847 | 0.4369 | 0.2699 | 0.4024 |
| 8 | 0.3883 | 1.0109 | 0.4458 | 0.2678 | 0.4116 |
| 9 | 0.3592 | 1.0327 | 0.4447 | 0.2634 | 0.4054 |
| 10 | 0.3259 | 1.0861 | 0.4121 | 0.2572 | 0.3782 |
| 11 | 0.2961 | 1.0954 | 0.4374 | 0.2572 | 0.4012 |
| 12 | 0.2694 | 1.1880 | 0.4151 | 0.2597 | 0.3840 |
| 13 | 0.2434 | 1.2165 | 0.4272 | 0.2658 | 0.3934 |
| 14 | 0.2196 | 1.2764 | 0.4244 | 0.2575 | 0.3913 |
| 15 | 0.1964 | 1.3218 | 0.4277 | 0.2623 | 0.3903 |
| 16 | 0.1768 | 1.3677 | 0.4177 | 0.2544 | 0.3840 |
| 17 | 0.1642 | 1.3979 | 0.4225 | 0.2586 | 0.3893 |

The recovered log's best semantic Macro-F1 is **0.445841 at zero-based epoch 8** (ninth recorded epoch). Best ROI-aggregated PUMA Macro-F1 is **0.277387 at zero-based epoch 6**. The final semantic Macro-F1 is **0.422517** and final validation loss **1.397857**. The training loss reaches **0.164161**, and LoRA gradient norm remains nonzero. The best semantic checkpoint and best PUMA checkpoint are different, so model selection must name the target estimand. The historical numbers are not directly comparable with the new pilots because splits, sampling units, coordinate policies and architectures differ.

The V17 documents report additional CPU surrogate observations: a simple baseline reached best fixed-ten Macro-F1 0.2538 versus 0.2307 for a phased pipeline; a KD phase degraded 0.2307 to 0.2083; C256 reached 0.2477 by epoch 20 while CU peaked at 0.2363 around epoch 7 then declined to 0.2127. Their six-ROI sample had only 2,738 nuclei, absent epithelium and only 2–3 nuclei in several other classes. Those are historical, document-reported results, not new replications and not real UNI2 performance. An apparent fusion gain failed a capacity/placebo comparison. This supports removing unjustified complexity from the next default; it does not prove a universal anti-fusion result.

The biology audit gives descriptive effect sizes, including H-gradient mean epsilon-squared about 0.289 and GT-neighbour entropy about 0.283. These full-data associations are not estimates of held-out incremental information given UNI2. Ground-truth neighbour classes are unavailable at inference. Several morphological descriptors were measured with annotated nuclear boundaries, whereas the current Stage-1 contract supplies coordinates only. Even a feature called “direct” in the old report is not automatically available under this narrower interface. No biology branch is added.

### 1.3 Literature synthesis

HoVer-Net and HoVer-NeXt establish strong joint nuclear segmentation/classification baselines, while CoNIC emphasizes that detection, segmentation, class assignment and counting require distinct evaluation. Reusing their published scores as a target for this conditional classifier would mix tasks and ontologies. PanNuke's five broad categories are not the ten PUMA phenotypes. NuCLS demonstrates why class definitions and interrater variation matter, especially when fine-grained immune identities are ambiguous on H&E. [HoVer-Net](https://github.com/vqdang/hover_net), [HoVer-NeXt](https://github.com/digitalpathologybern/hover_next_train), [CoNIC](https://arxiv.org/abs/2303.06274), [PanNuke](https://arxiv.org/abs/2003.10778), [NuCLS](https://pmc.ncbi.nlm.nih.gov/articles/PMC9112766/)

UNI2-h and CONCH motivate a pretrained-encoder baseline, but their broad pathology benchmarks do not establish ten-class nuclear separability at this crop scale. CellViT++ is a particularly relevant precedent for efficient cell-classifier adaptation using foundation features. This supports testing a small classifier before paying for encoder adaptation. It does not require replacing the fixed Stage-1 detector or fusing multiple foundation models. [UNI release and benchmarks](https://github.com/mahmoodlab/UNI), [CONCH](https://www.nature.com/articles/s41591-024-02856-4), [CellViT++](https://arxiv.org/abs/2501.05269)

Decoupled long-tail recognition motivates separating representation quality from classifier bias. Logit adjustment and Balanced Softmax offer simple explicit prior corrections; class-balanced focal loss is a legitimate alternate control but introduces additional hyperparameters. SupCon and Nuc2Vec motivate metric learning only when a representation bottleneck is demonstrated. Same-class positives should span independent cases; oversampling the same few nuclei is not equivalent to obtaining new biological variation. Accumulating separate minibatch losses does not by itself create a larger contrastive batch. [Decoupled classifiers](https://arxiv.org/abs/1910.09217), [Logit adjustment](https://arxiv.org/abs/2007.07314), [Balanced Softmax](https://proceedings.nips.cc/paper_files/paper/2020/hash/2ba61cc3a8f44143e1f2f13b2b729ab3-Abstract.html), [Class-balanced loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html), [SupCon](https://arxiv.org/abs/2004.11362), [Nuc2Vec](https://proceedings.mlr.press/v143/feng21a.html)

This survey covers the requested method families and nearby primary work. It is not a registered systematic review with database-complete inclusion/exclusion screening. No unverifiable citation or cross-dataset numerical claim is used to manufacture a SOTA conclusion.

## 2. Reviewer–architect dialectic and empirical record

The discussion below is an adversarial review of the investigation rather than an external peer review. Each decision is tied to observable evidence and a documented protocol. Protocol commits preceded corresponding experiments; amendments distinguish new information from post-result choices.

### Round 1 — The prevalence premise

**Reviewer:** “Your loss correction is built on an unsupported prior. Why should a text percentage describe the training fold?”

**Architect:** “It should not. The complete GeoJSON census and training split provide exact counts. The ratio text is retained as a conflicting source, not a loss constant.”

**Test:** parse all annotations; compare the ten counts with the biology CSV; count class-positive ROIs; inspect the four named image/annotation pairs. **Decision:** replace any hard-coded global ratio in training with the effective training sampler prior. **Failure found:** 290:1 versus 156:1 imbalance descriptions are not interchangeable.

### Round 2 — The encoder identity

**Reviewer:** “A filename is not model provenance. Did you actually run UNI2?”

**Architect:** “The first local file had a 1024-dimensional CLS and 16-pixel patch projection, so it was rejected as UNI2-h. The supplied corrected checkpoint has a 1536-dimensional CLS, eight registers, patch14 projection and a strictly matching state dictionary.”

**Test:** checkpoint tensor shapes, strict state loading, finite real forward output, and SHA256 recording. **Remediation:** install timm 1.0.20 under the workspace; the system timm 0.9.6 rejected `reg_tokens`. A meta-device constructor also failed because timm calls `.item()` on a schedule tensor; initialization skipping plus memory-mapped weight assignment fixed loading without changing the architecture. **Decision:** actual UNI2 features are used in the primary pilot; token-weight maps and original-UNI-shaped weights are never relabelled as UNI2 features.

### Round 3 — The fusion mechanism

**Reviewer:** “Where is your evidence that biology fusion starves minority gradients? Low F1 is not a gradient measurement.”

**Architect:** “There is no causal proof of that mechanism in the supplied records. In a concatenated classifier, the appearance gradient is `W_image^T(p-y)`. An easy competing feature can reduce residuals or change learned weights, and rare examples have small population mass. Neither implies every rare-class gradient vanishes.”

**Test:** inspect historical LoRA-gradient and per-class-recall fields; execute current positive-example gradient diagnostics. **Decision:** describe branch redundancy, sampling scarcity, prior mismatch and overfitting as distinct possibilities. The supplied records support early adaptation followed by overfitting. Remove branches under the present simplicity constraint; do not present removal as proof of a universal biological-fusion failure law.

### Round 4 — The learning criterion

**Reviewer:** “You demanded strictly descending loss. Stochastic augmentation and minibatching need not be monotonic. Would you discard a better classifier merely because one epoch increased?”

**Architect:** “No. We record the objective, unaugmented training loss, held-out NLL, Macro-F1, per-class recall, exposure and gradients. A finite downward trend is an optimization diagnostic; held-out performance decides usefulness.”

**Test:** ten-epoch CNN controls and ten-epoch UNI2 frozen-feature controls. All trajectories are retained. **Decision:** no stopping rule based solely on a single loss increase and no training extension solely to force monotonicity.

### Round 5 — Loss and sampler interaction

**Reviewer:** “Balanced sampling plus original-prior adjustment changes two things and may overcorrect the tail. What are you actually optimizing?”

**Architect:** “Three matched controls isolate this: natural CE, natural adjusted CE with tau=1, and balanced sampling with ordinary CE. The loss uses the sampler-induced prior. No focal, contrastive or extra weighted CE term is stacked onto the initial recipe.”

**Test:** analytical/autograd equivalence to Balanced Softmax; identical architecture, seed, epochs and learning rate in each pilot. **Decision:** keep the full control table below; any apparent winner remains a development choice and requires independent confirmation.

### Round 6 — Is this a representative sample?

**Reviewer:** “A few ROIs with no epithelium cannot settle a ten-class question. Neither can a deliberately balanced validation subset estimate population F1.”

**Architect:** “The new CNN screen samples up to 100 nuclei uniformly per ROI from all 205 ROIs. The CPU UNI2 screen deliberately covers all ten classes across groups, with additional random nuclei. Its enriched validation distribution is explicitly disclosed.”

**Test:** label and ROI supports, group disjointness and class-presence checks before training. **Decision:** the pilot can reject broken learning or gross collapse; it cannot certify SOTA or settle small tail-class differences. ROC/PR estimates and calibration from tiny enriched subsets cannot be sold as deployment measurements.

### Round 7 — Does LoRA earn its parameters?

**Reviewer:** “Why rank eight, why Q/V, why four blocks? Those are starting choices, not mathematical consequences of class frequencies.”

**Architect:** “Agreed. They create a restricted, falsifiable adaptation candidate: 196,608 adapter parameters, leaving the original backbone frozen. A paired five-epoch actual-UNI2 dynamics test is included. It is too small to establish an adaptation advantage.”

**Test:** zero-update equality at initialization; positive B gradients; expected zero A gradients until B changes; five-epoch real late-block adaptation versus an identical fresh frozen head. **Decision:** retain the frozen encoder as the least-complex reference until a larger matched experiment supports LoRA. Parameter count alone is not evidence of benefit.

**Measured outcome:** across the corrected five-epoch comparison, per-class single-example adapter-gradient norms range from 0.01350 to 0.52753. The two arms have identical validation confusion matrices at every epoch. Both reach training Macro-F1 1.0 while validation Macro-F1 reaches only 0.3038. This establishes an intact adaptation path and rapid memorization on this tiny sample; it provides no observed classification advantage from LoRA at the tested settings.

### Round 8 — What counts as convergence?

**Reviewer:** “You have repeated development-set feedback. Where is the independent evidence?”

**Architect:** “The investigation is not scientifically converged. It supplies executed mechanism tests, a minimal implementation and the next locked validation sequence. No repeated reuse of this validation fold is called an independent test.”

**Decision:** distinguish completion of this research deliverable from proof of SOTA. Publish only after the remaining validation in Section 4.5; do not manufacture convergence or keep adding components to make the report look complete.

### 2.1 Newly executed probe results

**Design and support.** The CNN uses a 48×48 resized version of the 96-pixel source crop and three small convolutional blocks, not UNI2. It includes 20,500 nuclei from all 205 ROIs. The real UNI2 pilot uses 220 training and 107 validation nuclei; its class-enriched validation spans 38 ROIs. Full feature extraction took 1,121 seconds. Its weights SHA256 is `32cc070acd809d087debdbe483f388561cf03de825cc7c2935d1732dfb427ffe`.

| Class | CNN train | CNN val | UNI2 train | UNI2 val |
| --- | --- | --- | --- | --- |
| tumor | 9591 | 2849 | 71 | 39 |
| lymphocyte | 3378 | 672 | 37 | 15 |
| plasma_cell | 81 | 14 | 14 | 6 |
| histiocyte | 1295 | 166 | 17 | 9 |
| melanophage | 114 | 66 | 12 | 8 |
| neutrophil | 49 | 37 | 12 | 5 |
| stroma | 709 | 149 | 16 | 6 |
| epithelium | 439 | 65 | 14 | 7 |
| endothelium | 320 | 85 | 13 | 6 |
| apoptosis | 324 | 97 | 14 | 6 |

**CNN controls, one seed, ten epochs each.**

| Recipe | Best val F1 | Epoch | Final val F1 | Best balanced accuracy | Best NLL | Best ECE15 |
| --- | --- | --- | --- | --- | --- | --- |
| balanced_ce | 0.2899 | 8 | 0.2702 | 0.4060 | 1.3974 | 0.0305 |
| ce | 0.2211 | 6 | 0.2178 | 0.2298 | 0.9272 | 0.0450 |
| la | 0.2695 | 8 | 0.2669 | 0.4367 | 1.6573 | 0.0460 |

At epoch 10, natural CE gives zero recall to plasma_cell, neutrophil and epithelium. The adjusted and balanced-sampler controls have nonzero recall in all ten classes. All classes have positive training-example gradients. This is evidence that zero recall can coexist with an intact gradient path; it is not proof that balanced sampling is universally better. The sample is uniform within each ROI with a 100-nucleus cap, so it is not an exact global-prevalence sample.

**Real frozen UNI2, three seeds, ten epochs per run.**

| Recipe / seed | Best val F1 | Epoch | Final val F1 | Final train F1 | Best balanced accuracy | Best ECE15 |
| --- | --- | --- | --- | --- | --- | --- |
| balanced_ce | 0.3913 | 9 | 0.3869 | 0.9027 | 0.4201 | 0.1164 |
| balanced_ce_seed29 | 0.4378 | 8 | 0.4105 | 0.8946 | 0.4761 | 0.1286 |
| balanced_ce_seed43 | 0.4194 | 10 | 0.4194 | 0.9095 | 0.4451 | 0.1166 |
| ce | 0.4416 | 10 | 0.4416 | 0.9733 | 0.4152 | 0.1416 |
| ce_seed29 | 0.3982 | 8 | 0.3860 | 0.9771 | 0.3681 | 0.1124 |
| ce_seed43 | 0.4390 | 8 | 0.4133 | 0.9862 | 0.4170 | 0.1758 |
| la | 0.4497 | 9 | 0.4356 | 0.9357 | 0.4859 | 0.1190 |
| la_seed29 | 0.4599 | 10 | 0.4599 | 0.9477 | 0.4925 | 0.1007 |
| la_seed43 | 0.4508 | 9 | 0.4115 | 0.9432 | 0.4909 | 0.1478 |

| Recipe | Best-checkpoint mean ± sample SD | Final-epoch mean ± sample SD |
| --- | --- | --- |
| ce | 0.4262 ± 0.0243 | 0.4136 ± 0.0278 |
| la | 0.4535 ± 0.0056 | 0.4357 ± 0.0242 |
| balanced_ce | 0.4162 ± 0.0234 | 0.4056 ± 0.0168 |

The mean paired best-checkpoint LA-minus-CE difference is **+0.0272**. The exploratory 2,000-replicate ROI-bootstrap interval is **[-0.0497, +0.1401]**. It includes zero. This interval conditions on development-selected checkpoints and an enriched subset; it is not a confirmatory confidence interval for deployment superiority. Use natural adjusted CE as a provisional candidate and retain ordinary CE as the matched reference; do not declare the loss question settled.

**Classwise F1 / recall at seed17 development-selected UNI2 checkpoints.**

| Class | Val support | CE F1 / recall | LA F1 / recall | Balanced CE F1 / recall |
| --- | --- | --- | --- | --- |
| tumor | 39 | 0.6602 / 0.8718 | 0.6914 / 0.7179 | 0.6905 / 0.7436 |
| lymphocyte | 15 | 0.4286 / 0.4000 | 0.4348 / 0.3333 | 0.2963 / 0.2667 |
| plasma_cell | 6 | 0.5000 / 0.3333 | 0.4000 / 0.3333 | 0.4706 / 0.6667 |
| histiocyte | 9 | 0.0000 / 0.0000 | 0.1429 / 0.1111 | 0.1250 / 0.1111 |
| melanophage | 8 | 0.4286 / 0.3750 | 0.4348 / 0.6250 | 0.3529 / 0.3750 |
| neutrophil | 5 | 0.7500 / 0.6000 | 0.6667 / 1.0000 | 0.6154 / 0.8000 |
| stroma | 6 | 0.5333 / 0.6667 | 0.4444 / 0.6667 | 0.4615 / 0.5000 |
| epithelium | 7 | 0.6154 / 0.5714 | 0.6154 / 0.5714 | 0.6154 / 0.5714 |
| endothelium | 6 | 0.5000 / 0.3333 | 0.6667 / 0.5000 | 0.2857 / 0.1667 |
| apoptosis | 6 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 |

Apoptosis recall is zero at all three selected seed17 checkpoints, despite positive training head gradients and high training Macro-F1. Histiocyte is also zero for the CE selection. The pilot does not meet the requested all-class generalization/convergence standard. Further encoder adaptation is not automatically the remedy.

**Exploratory alternate-boundary diagnostic after the zero-recall finding.** Training-only cosine centroids and k-nearest neighbours were evaluated without validation-fitted parameters.

| Classifier | Val Macro-F1 | Apoptosis recall |
| --- | --- | --- |
| centroid | 0.3727 | 0.0000 |
| knn1 | 0.2434 | 0.0000 |
| knn5 | 0.3142 | 0.0000 |

Similarity/vote scores are not calibrated probabilities. This diagnostic is exploratory, and nearest-neighbour vote ties use the lowest class index. It does not establish a new production winner.

**Actual UNI2 restricted-LoRA dynamics, five epochs.** Thirty training nuclei (three per class) and twenty validation nuclei (two per class) use fixed GT-centred crops and the same original ROI split. Frozen prefix outputs are cached before block20. The corrected comparison uses identical fresh head initialization and an independent seed17 minibatch generator. Preliminary logs from a different-shuffle comparison were preserved but excluded.

**frozen:**

| Epoch | Train objective | Train Macro-F1 | Val Macro-F1 | Val NLL | Zero-recall val classes |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.8682 | 0.4757 | 0.1333 | 2.2370 | tumor, lymphocyte, plasma_cell, neutrophil, stroma, epithelium, endothelium, apoptosis |
| 2 | 1.3978 | 0.9250 | 0.1686 | 2.1745 | lymphocyte, plasma_cell, histiocyte, epithelium, endothelium, apoptosis |
| 3 | 0.6452 | 1.0000 | 0.1357 | 2.0603 | tumor, lymphocyte, plasma_cell, histiocyte, stroma, endothelium, apoptosis |
| 4 | 0.2787 | 1.0000 | 0.2786 | 1.9787 | lymphocyte, plasma_cell, histiocyte, endothelium, apoptosis |
| 5 | 0.1019 | 1.0000 | 0.3038 | 2.0618 | lymphocyte, plasma_cell, histiocyte, endothelium, apoptosis |

**lora:**

| Epoch | Train objective | Train Macro-F1 | Val Macro-F1 | Val NLL | Zero-recall val classes |
| --- | --- | --- | --- | --- | --- |
| 1 | 2.8682 | 0.4757 | 0.1333 | 2.2370 | tumor, lymphocyte, plasma_cell, neutrophil, stroma, epithelium, endothelium, apoptosis |
| 2 | 1.3975 | 0.9250 | 0.1686 | 2.1744 | lymphocyte, plasma_cell, histiocyte, epithelium, endothelium, apoptosis |
| 3 | 0.6448 | 1.0000 | 0.1357 | 2.0603 | tumor, lymphocyte, plasma_cell, histiocyte, stroma, endothelium, apoptosis |
| 4 | 0.2784 | 1.0000 | 0.2786 | 1.9786 | lymphocyte, plasma_cell, histiocyte, endothelium, apoptosis |
| 5 | 0.1017 | 1.0000 | 0.3038 | 2.0617 | lymphocyte, plasma_cell, histiocyte, endothelium, apoptosis |

This small dynamics check is not a full-data LoRA comparison. Positive adapter gradients, when reported, verify connectivity, not useful representation change. No adaptation advantage is promoted from twenty validation nuclei.

**Complete ten-epoch trajectories (objective / validation Macro-F1).**

**cnn, seed17:**

| Epoch | CE objective / F1 | LA objective / F1 | Balanced CE objective / F1 |
| --- | --- | --- | --- |
| 1 | 1.1910 / 0.1320 | 1.1171 / 0.1428 | 1.8957 / 0.1499 |
| 2 | 1.0039 / 0.1601 | 0.9764 / 0.1855 | 1.5375 / 0.2065 |
| 3 | 0.9498 / 0.1692 | 0.9209 / 0.1985 | 1.4010 / 0.1997 |
| 4 | 0.9185 / 0.1754 | 0.8915 / 0.2259 | 1.3293 / 0.2098 |
| 5 | 0.8895 / 0.1737 | 0.8661 / 0.2445 | 1.2653 / 0.2568 |
| 6 | 0.8643 / 0.2211 | 0.8405 / 0.2201 | 1.2229 / 0.2724 |
| 7 | 0.8479 / 0.2001 | 0.8233 / 0.2215 | 1.1755 / 0.2582 |
| 8 | 0.8441 / 0.1922 | 0.8144 / 0.2695 | 1.1255 / 0.2899 |
| 9 | 0.8230 / 0.2104 | 0.7966 / 0.2637 | 1.0804 / 0.2653 |
| 10 | 0.8075 / 0.2178 | 0.7839 / 0.2669 | 1.0513 / 0.2702 |

**uni2, seed17:**

| Epoch | CE objective / F1 | LA objective / F1 | Balanced CE objective / F1 |
| --- | --- | --- | --- |
| 1 | 2.2978 / 0.0531 | 2.1673 / 0.2015 | 2.3649 / 0.0537 |
| 2 | 1.6471 / 0.3292 | 1.5802 / 0.3523 | 1.7994 / 0.3067 |
| 3 | 1.3032 / 0.3591 | 1.2288 / 0.3550 | 1.3498 / 0.3047 |
| 4 | 1.0253 / 0.3264 | 0.9835 / 0.4079 | 1.0692 / 0.3738 |
| 5 | 0.8509 / 0.4175 | 0.8093 / 0.4112 | 0.8170 / 0.3613 |
| 6 | 0.6959 / 0.4170 | 0.6656 / 0.4013 | 0.7990 / 0.3511 |
| 7 | 0.5895 / 0.4039 | 0.5586 / 0.4166 | 0.6143 / 0.3492 |
| 8 | 0.4917 / 0.3888 | 0.4694 / 0.4333 | 0.5430 / 0.3768 |
| 9 | 0.4079 / 0.4196 | 0.3874 / 0.4497 | 0.4887 / 0.3913 |
| 10 | 0.3473 / 0.4416 | 0.3342 / 0.4356 | 0.4102 / 0.3869 |

Training objectives use different effective priors/samplers and must not be ranked as if they were identical risk functions. The archived JSON files include all epochs for all seeds, class exposures, full confusion matrices, unaugmented training metrics and classwise gradient probes.

![Historical learning curve](../../../SHARED/IMAGES/FIGURES/history.png)

![CNN controls](../../../SHARED/IMAGES/FIGURES/cnn.png)

![UNI2 seed replications](../../../SHARED/IMAGES/FIGURES/uni2.png)

![Corrected LoRA dynamics](../../../SHARED/IMAGES/FIGURES/lora.png)


### 2.2 Integrity checks and limitations

| Check | Outcome |
| --- | --- |
| edge_center_preserved | PASS |
| balanced_softmax_and_gradient | PASS |
| lora_zero_B_initialization_and_delayed_A_gradient | PASS |
| partial_accumulation_equivalence | PASS |
| fixed10_missing_classes | PASS |
| wrong_checkpoint_rejected | PASS |
| absolute_relative_image_leakage_rejected | PASS |

The contact sheet was visually reviewed for coordinate orientation, target centring and class variety. This verifies gross data alignment, not the medical correctness of every label. A pathologist review of systematically confused classes remains an independent scientific requirement. Test outputs and raw per-epoch JSON files are included in the reproducibility bundle.

## 3. Minimal architecture and mathematical specification

### 3.1 Input and encoder

For nucleus `i`, read RGB image `I` and the fixed coordinate `(x_i,y_i)`. Extract one square crop of side `s=96` source pixels; pad missing border context white without shifting the centre. Resize to 224×224 with bicubic interpolation. The 96-pixel setting is the preregistered pilot field of view, not a demonstrated optimum. Larger crops contain additional cells and reduce the target's relative size when resized; smaller crops may truncate informative nuclear or perinuclear structure. Screen 64, 96 and 128 source pixels under the same encoder and split before considering dual-scale input. No dual-scale branch is promoted by these pilots.

The source crop covers roughly 21.7 micrometres at inspected TIFF resolution. Resizing is a model input transform, not new optical information. Train augmentation in the image-based implementation is random rotations by multiples of 90 degrees plus reflection. These retain the target near the centre. Frozen-feature and cached-prefix pilots use fixed views so their caching remains valid. Stain perturbation is off by default. An optional small RGB optical-density scaling is available and correctly named; it is not Macenko or Vahadane normalization and is unvalidated here. No per-class stain transform, hard stain normalization, mask channel, tissue map or label-derived context is introduced.

Normalize with RGB mean `(0.485,0.456,0.406)` and standard deviation `(0.229,0.224,0.225)`. UNI2-h uses 224 input, patch size 14, width 1536, depth 24, 24 attention heads, eight register tokens, no positional embedding for class/register tokens, SwiGLUPacked MLP with ratio `2.66667*2`, SiLU and layer-scale initialization `1e-5`. Read the CLS representation; do not silently average registers or concatenate token pools. [Official UNI2-h model definition](https://huggingface.co/MahmoodLab/UNI2-h)

### 3.2 Head and optional adapter

Let `h_i=f_theta(x_i)` be the 1536-dimensional CLS vector. The head is

\[
u_i=\operatorname{LayerNorm}_{\text{no affine}}(h_i),\qquad z_i=Wu_i+b,\quad W\in\mathbb R^{10\times1536}.
\]

The normalization is per vector, with no train/validation population fitting and no trainable affine parameters. The linear head has **15,370 trainable parameters**. There is no learned projection MLP, cosine margin, prototype bank, auxiliary decoder or graph. Feature normalization itself remains an explicit control choice; do not claim it was proven superior to raw CLS by the present experiments.

For optional LoRA on blocks 20–23 (zero-based), update Q and V separately inside timm's fused QKV projection:

\[
Q'=Q+\frac{\alpha}{r}X A_Q^T B_Q^T,\qquad
V'=V+\frac{\alpha}{r}X A_V^T B_V^T.
\]

Use `r=8`, `alpha=16`, no LoRA dropout, random Kaiming A and zero B. K, attention output projections, MLPs, original normalization weights, patch embedding and register tokens remain frozen. Each Q or V update costs `2dr`; four blocks and two projections cost `4*4*1536*8 = 196,608` parameters. Including the head gives **211,978**. LoRA A has zero initial gradient when B is zero; this is expected initialization behavior, not a broken adapter. [LoRA](https://arxiv.org/abs/2106.09685)

Feature geometry and long-tail adaptation interact through the data-weighted gradient. A class with few independent nuclei contributes fewer positive update directions; a low-rank adapter can only express updates through its learned factors. Strong global pathology features do not guarantee discrimination between nearby nuclear phenotypes. Conversely, a boundary/prior error can exist even when features separate those classes. The exploratory geometry measurements below describe this pilot's feature sample; they do not establish a universal UNI2 collapse, because empirical covariance rank is bounded by the sample count.

The empirical training-feature covariance effective rank is **102.24**, with maximum sample rank **219** for 220 training points. Positive class-centroid cosines are high, but this alone is not proof of collapse: a common component and small sample affect cosine geometry. No PCA/whitening was selected from validation or inserted into the production model.

### 3.3 Loss and inference semantics

Let `q_k` denote the class prior induced by the **actual training sampler**. With natural sampling, `q_k=n_k/N` for the training split or training subsample. With exact inverse-frequency sampling, the expected `q` is uniform. Define

\[
\mathcal L_{\tau}(z,y)=-\log\frac{\exp(z_y+\tau\log q_y)}{\sum_{k=1}^{10}\exp(z_k+\tau\log q_k)}.
\]

Natural CE uses tau=0. Natural adjusted CE uses tau=1. At tau=1, replacing normalized priors by counts adds the same constant to every logit, so this is Balanced Softmax. The plus sign is intentional. Under the idealized label-shift model and unrestricted population-risk minimization,

\[
z_k(x)=\log p(x\mid y=k)+(1-\tau)\log q_k+C(x).
\]

Thus for tau=1 the unadjusted raw `z` is used for a balanced-prior decision rule. For an ordinary CE model, the separate post-hoc balanced correction is `z-log(q)`. Do not train with tau=1 and then subtract the same prior again. For a specified target prior `rho`, population-probability logits are `z-(1-tau)log(q)+log(rho)`. These identities assume stable class-conditionals and sufficient model fit; real finite-data calibration must still be evaluated. [Training and post-hoc logit adjustment](https://arxiv.org/abs/2007.07314)

The per-example derivative is `dL/dz_k = softmax(z+tau log(q))_k - 1[k=y]`. A missing positive class in a batch still receives negative-class gradients. A nonzero gradient does not guarantee a correctly placed decision boundary, and balanced-error consistency is not a theorem of Macro-F1 optimality.

For a future class-balanced focal control, the effective-number weight is proportional to `(1-beta)/(1-beta^n_k)` and the focal term is `(1-p_y)^gamma`. It should replace, not silently accompany, the selected initial loss/sampler control. Beta and gamma must be prospectively chosen and validated. SupCon would add another objective, a pair-selection policy and a temperature; its coefficient remains zero in this package because incremental benefit has not been established. [Class-balanced loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html), [Supervised contrastive learning](https://arxiv.org/abs/2004.11362)

### 3.4 Calibration

Report raw and calibrated NLL, ECE15, reliability diagrams and classwise performance on a separate naturally sampled calibration/evaluation population. Fit a single positive temperature only after model selection; it preserves argmax, so it cannot rescue zero tail recall. The supplied implementation searches temperature from 0.1 to 10 on an explicit calibration file. If the optimum hits a bound, investigate and report it. Do not fit this transformation on the tiny enriched pilot and call it deployment calibration. [Temperature scaling](https://proceedings.mlr.press/v70/guo17a.html)

## 4. Execution blueprint and hyperparameters

### 4.1 Locked sequence

1. Freeze Stage-1 checkpoint hashes, coordinate frame, decoder settings and output IDs. Verify patient/slide/ROI identity mapping and preserve the intended official split. The new package does not retrain Stage 1.
2. Audit annotations, class names, duplicates, missing images, TIFF scale, train/validation group overlap and per-class case support. Fit no stain statistics, PCA, prior or calibration parameters on test data.
3. Run a clean GT-centred semantic control and a matched frozen-Stage-1-centred control using one-to-one geometry-only supervision. Evaluate the same matched nuclei for the centring comparison. Also retain all proposals and missed GT for the separate end-to-end evaluator.
4. Extract verified frozen UNI2 CLS features with fixed preprocessing; compare the three loss/sampling controls. Then compare source FOVs while changing one factor. Use development data only; do not endlessly mine an untouched test set.
5. Warm up a head on the selected representation, then run a matched frozen versus restricted-LoRA ablation. If initializing from a warm head, initialize both arms from that same head. The supplied fresh-head CLI is also a valid paired control; record which protocol is used.
6. Repeat candidate improvements across patient folds and seeds, with per-class failure analysis and paired group-level uncertainty. Promote additional parameters only if they improve the primary endpoint reproducibly at acceptable cost.
7. Fit calibration on a dedicated group; run one locked final evaluation and the external official matcher. Keep the original Stage-1 score and coordinates fixed when isolating classification effects.

### 4.2 Starting hyperparameters, not claimed optima

| Setting | Frozen head | Restricted LoRA |
|---|---|---|
| Encoder | verified UNI2-h, frozen | original weights frozen; Q/V adapters in blocks 20–23 |
| Input | source FOV 96, resize 224 | identical |
| Head | parameter-free LayerNorm + Linear(1536,10) | identical |
| Optimizer | AdamW, betas (0.9,0.999), epsilon 1e-8 | identical |
| Head peak LR | 1e-3 | 1e-3 |
| Adapter peak LR | none | 1e-5 |
| Weight decay | 0.01 for matrices; 0 for vector/bias | same rule |
| Effective batch | 64 | 64 |
| Example microbatch / accumulation | 8 / 8 | 8 / 8; reduce microbatch if necessary |
| Schedule in image-based CLI | 5% optimizer-step warmup; cosine to 10% peak | identical |
| Maximum initial full-data screen | 20 epochs | 20 epochs |
| Early stopping | validation Macro-F1 patience 5, improvement threshold 1e-4 | identical |
| Precision | FP32 CPU; optional GPU autocast | BF16 on supported CUDA, otherwise scaled FP16 |
| Gradient clipping | global norm 1 | global norm 1 |
| LoRA rank / alpha | absent | 8 / 16 |
| Stain augmentation | off; optional perturbation is an ablation | same |

The executed lightweight pilots used constant learning rates for their short fixed epochs, as stated in their protocols. They are not evidence that the full-data warmup/cosine schedule is optimal. AdamW is chosen for a conventional reproducible baseline; no ScheduleFree advantage was tested. Accumulation divides summed gradients by the actual number of samples in each window, including a partial final window. It does not create additional unique rare examples.

### 4.3 Metrics and inferential discipline

Primary development endpoint: fixed-ten-class Macro-F1, `mean_k 2TP_k/(2TP_k+FP_k+FN_k)`, with zero divisions explicitly handled. Always report per-class precision, recall, F1 and support. Balanced accuracy is mean recall over supported classes; a separately labelled fixed-ten variant assigns absent-class recall zero. A split missing a class cannot establish ten-class performance even if a software metric returns a number.

Report the 10×10 confusion matrix, predicted/true prevalence ratio, one-versus-rest PR-AUC on adequately supported independent data, NLL, ECE15 and classwise calibration. Separate source domains (primary/metastatic) and patient groups where available. For comparisons, bootstrap patients/ROIs as clusters, not individual nuclei; preserve pairing between models. Wide rare-class intervals are a data limitation, not a prompt to find a more favorable random seed. Repeated development-set selection invalidates a naive confirmatory p-value; the bootstrap attached to the pilot is exploratory.

Official PUMA ROI-averaged and pooled TP/FP/FN scores must be named separately. Conditional classification Macro-F1 at GT centres excludes missed and spurious detections. A proposal-level semantic count can differ from a unique-GT count due to duplicates or resolver admission; no numerical equality is assumed. The package preserves original proposal scores rather than claiming to reproduce the official class-conditioned matching code.

### 4.4 Failure-specific next decisions

| Observed pattern | Required check | Permitted next move |
|---|---|---|
| Positive examples rarely sampled | exposure and independent ROI coverage | sampler control; avoid repeated copies as a substitute for new cases |
| Nonzero gradients but zero recall | confusers, margins, class probability mass | classifier/prior comparison before encoder expansion |
| Train improves, validation degrades | grouped curves and seeds | earlier checkpoint or reduce adaptation; not more epochs by default |
| Good ranking, bad argmax | cross-fitted decision bias, proper prior accounting | calibration/decision analysis with separate selection data |
| Poor separability across folds | FOV, label audit, real encoder comparison | restricted adaptation or different single encoder |
| Improvement only in richer branch | matched capacity/placebo and group controls | no branch promotion without unique validated gain |

### 4.5 Conditions still required for a publication-level claim

The proposed pipeline must improve a matched full-data reference on locked patient groups, with multiple seeds and a justified uncertainty analysis. Every supported class needs meaningful recall and error review; the current five-to-six validation neutrophils in the real-UNI2 pilot cannot settle tail robustness. Deployable frozen-Stage-1 coordinates must be tested, since GT-centred crops remove real localisation noise. The full ten-class external benchmark and its exact scorer must be run without further tuning. Natural-prevalence calibration, source-domain shifts and possible pretraining-data overlap need assessment. These requirements are not yet satisfied, so neither “fully converged” nor “SOTA” is a valid result label.

## 5. Self-contained PyTorch implementation and reproducibility

The deliverable contains `dataset.py`, `model.py`, `loss.py` and `train_eval.py`, plus `predict.py`, `attach_stage1.py`, `calibrate.py`, `requirements.txt` and execution instructions in `README.md`. The research scripts, protocols, sample manifests and complete small-result logs are included in the accompanying archive. Raw images and pretrained weights are not duplicated. The source dataset and Stage 1 were not edited.

The core implementation is reproduced below so this Markdown report can be read independently. Run commands and provenance requirements are documented in `README.md`. The package is a tested research implementation, not a clinically validated product. CPU numerical/integrity checks and the explicitly listed pilots were executed; CUDA execution and full-scale deployment remain untested here.

### dataset.py

```python
"""Stage-2 crops only. All coordinates are level-0 ROI pixel x,y; Stage 1 is external."""
import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

CLASSES = ['tumor','lymphocyte','plasma_cell','histiocyte','melanophage',
           'neutrophil','stroma','epithelium','endothelium','apoptosis']
MEAN = torch.tensor([.485,.456,.406])[:,None,None]
STD = torch.tensor([.229,.224,.225])[:,None,None]

def read_manifest(path):
    path = Path(path)
    with path.open(newline='',encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = {'uid','roi','group','image','x','y','label','split','coordinate_source'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifest must contain {sorted(required)}')
    # Prefer human-readable class names; existing manifests require an explicit sidecar.
    if 'class_name' not in rows[0]:
        sidecar=path.with_suffix('.json')
        if not sidecar.exists() or json.loads(sidecar.read_text()).get('classes')!=CLASSES:
            raise ValueError('Manifest requires class_name column or canonical classes in JSON sidecar')
    seen = set(); group_split = {}; roi_split = {}; image_split = {}
    for r in rows:
        if r['uid'] in seen: raise ValueError('Duplicate uid: '+r['uid'])
        seen.add(r['uid'])
        if not r['image']: raise ValueError('Empty image')
        p = Path(r['image'])
        r['image'] = str((p if p.is_absolute() else path.parent/p).resolve())
        for key, ledger in [('group',group_split),('roi',roi_split),('image',image_split)]:
            if not r[key]: raise ValueError(f'Empty {key}')
            value = str(Path(r[key])).casefold() if key=='image' else r[key]
            if value in ledger and ledger[value] != r['split']:
                raise ValueError(f'{key} leakage: {r[key]}')
            ledger[value] = r['split']
        if r['split'] not in {'train','val','calibration','test','predict'}:
            raise ValueError('Unknown split')
        if r['coordinate_source'] not in {'gt','stage1_oof','stage1_frozen'}:
            raise ValueError('Unknown coordinate provenance')
        r['label'] = int(r['label']); r['x'] = float(r['x']); r['y'] = float(r['y'])
        if r['label'] not in range(10) and not (r['split']=='predict' and r['label']==-1):
            raise ValueError('Label outside canonical ten-class ontology')
        if 'class_name' in r and r['label']>=0 and r['class_name']!=CLASSES[r['label']]:
            raise ValueError('Integer label and class name disagree')
        if not np.isfinite([r['x'],r['y']]).all(): raise ValueError('Nonfinite center')
        if not Path(r['image']).is_file(): raise FileNotFoundError(r['image'])
    return rows

def centered_crop(image, x, y, fov):
    """Preserve requested center near edges; pad missing context white, never shift inward."""
    if fov < 2: raise ValueError('fov must be >=2 pixels')
    w,h = image.size
    if not (0 <= x < w and 0 <= y < h): raise ValueError('Center outside ROI')
    left,top = int(np.floor(x-fov/2+.5)),int(np.floor(y-fov/2+.5))
    box=(max(left,0),max(top,0),min(left+fov,w),min(top+fov,h))
    out=Image.new('RGB',(fov,fov),(255,255,255))
    out.paste(image.crop(box),(box[0]-left,box[1]-top))
    return out

def image_tensor(crop, output_size=224, augment=False, stain_strength=0.):
    crop=crop.resize((output_size,output_size),Image.Resampling.BICUBIC)
    a=np.array(crop,dtype=np.float32)/255.
    if augment:
        a=np.rot90(a,int(torch.randint(4,()).item()))
        if torch.rand(())<.5: a=a[:,::-1]
        if stain_strength:
            # Optional RGB optical-density perturbation; not stain deconvolution.
            scales=1+stain_strength*(2*torch.rand(3).numpy()-1)
            a=np.exp(np.log(np.clip(a,1/255,1))*scales)
    t=torch.from_numpy(a.copy()).permute(2,0,1)
    return (t-MEAN)/STD

class NucleiDataset(Dataset):
    def __init__(self, rows, fov=96, output_size=224, augment=False, stain_strength=0., cache_rois=2):
        self.rows=rows; self.fov=fov; self.output_size=output_size
        self.augment=augment; self.stain_strength=stain_strength
        self.cache_rois=cache_rois; self.cache=OrderedDict()
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; path=r['image']
        if path not in self.cache:
            with Image.open(path) as im: self.cache[path]=im.convert('RGB')
            while len(self.cache)>self.cache_rois: self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        crop=centered_crop(self.cache[path],r['x'],r['y'],self.fov)
        return image_tensor(crop,self.output_size,self.augment,self.stain_strength),r['label'],i

def sampling(rows, mode='natural', seed=17):
    labels=torch.tensor([r['label'] for r in rows])
    counts=torch.bincount(labels,minlength=10).double()
    if (counts==0).any(): raise ValueError('Training lacks classes: '+str([CLASSES[i] for i in range(10) if counts[i]==0]))
    if mode=='natural': return None,counts/counts.sum()
    if mode!='balanced': raise ValueError(mode)
    weights=1/counts[labels]
    q=torch.zeros(10,dtype=torch.float64).scatter_add_(0,labels,weights); q/=q.sum()
    return WeightedRandomSampler(weights,len(rows),replacement=True,generator=torch.Generator().manual_seed(seed)),q

def make_manifest(root, out, group_csv=None):
    """GT centers are a clean semantic control; never claim them as Stage-1 detections."""
    from shapely.geometry import shape
    from sklearn.model_selection import StratifiedGroupKFold
    root=Path(root); rows=[]
    groups={}
    if group_csv:
        with open(group_csv,newline='',encoding='utf-8') as f:
            groups={r['roi']:r['patient_id'] for r in csv.DictReader(f)}
    for p in sorted((root/'01_training_dataset_geojson_nuclei').glob('*.geojson')):
        roi=p.stem.removesuffix('_nuclei'); image=root/'01_training_dataset_tif_ROIs'/(roi+'.tif')
        if not image.is_file(): raise FileNotFoundError(image)
        if group_csv and roi not in groups: raise ValueError('Missing patient mapping: '+roi)
        obj=json.loads(p.read_text()); features=obj['features'] if isinstance(obj,dict) else obj
        for i,f in enumerate(features):
            name=f['properties']['classification']['name'].removeprefix('nuclei_')
            if name not in CLASSES: raise ValueError(name)
            g=shape(f['geometry'])
            if g.is_empty or not g.is_valid or g.area<=0: raise ValueError(f'Invalid nucleus polygon {roi}/{i}')
            center=g.centroid
            rows.append(dict(uid=f'{roi}:{i}',roi=roi,group=groups.get(roi,roi),image=str(image.resolve()),
                             x=center.x,y=center.y,label=CLASSES.index(name),class_name=name,split='',coordinate_source='gt'))
    if not rows: raise ValueError('No GeoJSON nuclei found')
    # A fixed, reproducible development split. This does not preserve legacy fold membership.
    split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=17)
    fold=np.zeros(len(rows),dtype=int)
    for k,(_,ix) in enumerate(split.split(np.zeros(len(rows)),[r['label'] for r in rows],[r['group'] for r in rows])):
        fold[ix]=k
    for r,k in zip(rows,fold): r['split']='val' if k==0 else 'train'
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta={'classes':CLASSES,'grouping':'patient' if group_csv else 'ROI ONLY; patient separation unverified',
          'source':'GT-centered semantic development control','seed':17,
          'counts':{s:np.bincount([r['label'] for r in rows if r['split']==s],minlength=10).tolist() for s in ['train','val']}}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--group-csv')
    a=p.parse_args();make_manifest(a.root,a.out,a.group_csv)

```

### model.py

```python
"""Strict local UNI2-h loader and minimal Q,V LoRA. No network downloads."""
import math,hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F

def checkpoint_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def load_uni2(path):
    import timm
    # Memory-map checkpoint tensors and assign them after skipping initialization.
    state=torch.load(Path(path),map_location='cpu',weights_only=True,mmap=True)
    if 'state_dict' in state: state=state['state_dict']
    expected={'cls_token':(1,1,1536),'reg_token':(1,8,1536),'patch_embed.proj.weight':(1536,3,14,14)}
    for key,shape in expected.items():
        if key not in state or tuple(state[key].shape)!=shape:
            raise ValueError(f'Not a UNI2-h checkpoint: expected {key} {shape}')
    if not hasattr(timm.layers,'SwiGLUPacked'): raise RuntimeError('Install timm==1.0.20')
    kw=dict(img_size=224,patch_size=14,depth=24,num_heads=24,init_values=1e-5,
            embed_dim=1536,mlp_ratio=2.66667*2,num_classes=0,no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,act_layer=nn.SiLU,reg_tokens=8,dynamic_img_size=True)
    # timm 1.0.20 calls .item() on linspace during construction, so a global
    # meta-device context is unsupported. Skip initialization and assign mmap weights.
    encoder=timm.create_model('vit_giant_patch14_224',pretrained=False,weight_init='skip',**kw)
    encoder.load_state_dict(state,strict=True,assign=True)
    return encoder

class QVLoRA(nn.Module):
    """Separate rank-r updates for Q and V in timm's fused QKV; K remains frozen."""
    def __init__(self,base,rank=8,alpha=16):
        super().__init__();self.base=base;self.rank=rank;self.scale=alpha/rank
        d=base.in_features
        if base.out_features!=3*d or rank<1: raise ValueError('Expected fused 3D x D QKV')
        self.base.requires_grad_(False)
        self.aq=nn.Parameter(torch.empty(rank,d));self.av=nn.Parameter(torch.empty(rank,d))
        self.bq=nn.Parameter(torch.zeros(d,rank));self.bv=nn.Parameter(torch.zeros(d,rank))
        nn.init.kaiming_uniform_(self.aq,a=math.sqrt(5));nn.init.kaiming_uniform_(self.av,a=math.sqrt(5))
    def forward(self,x):
        q,k,v=self.base(x).chunk(3,dim=-1)
        q=q+self.scale*F.linear(F.linear(x,self.aq),self.bq)
        v=v+self.scale*F.linear(F.linear(x,self.av),self.bv)
        return torch.cat([q,k,v],dim=-1)

class Stage2(nn.Module):
    def __init__(self,weights,mode='linear',rank=8,alpha=16,last_blocks=4,checkpointing=False):
        super().__init__()
        if mode not in {'linear','lora'}: raise ValueError(mode)
        self.encoder=load_uni2(weights);self.encoder.requires_grad_(False);self.mode=mode
        # Construct the common head before adapter initialization consumes RNG draws.
        self.head=nn.Linear(1536,10)
        if mode=='lora':
            if not 1<=last_blocks<=24: raise ValueError('last_blocks outside [1,24]')
            for b in self.encoder.blocks[-last_blocks:]: b.attn.qkv=QVLoRA(b.attn.qkv,rank,alpha)
            if checkpointing:self.encoder.set_grad_checkpointing(True)
    def features(self,x):
        # Parameter-free standardization within each vector; no statistics fit on validation.
        return F.layer_norm(self.encoder(x).float(),(1536,))
    def forward(self,x):
        if self.mode=='linear':
            with torch.no_grad(): h=self.features(x)
        else: h=self.features(x)
        return self.head(h)
    def train(self,mode=True):
        super().train(mode)
        if self.mode=='linear': self.encoder.eval()
        return self

```

### loss.py

```python
"""Training prior and inference prior are explicit and separate."""
import torch
from torch import nn
from torch.nn import functional as F

class LogitAdjustedCE(nn.Module):
    def __init__(self, sampling_prior, tau=1.):
        super().__init__()
        p=torch.as_tensor(sampling_prior,dtype=torch.float32)
        if p.shape!=(10,) or not torch.isfinite(p).all() or (p<=0).any():
            raise ValueError('All ten training/sampling priors must be positive and finite')
        if not 0<=tau<=2: raise ValueError('tau outside supported [0,2]')
        self.register_buffer('log_prior',(p/p.sum()).log());self.tau=float(tau)
    def forward(self,logits,labels,reduction='mean'):
        return F.cross_entropy(logits.float()+self.tau*self.log_prior,labels,reduction=reduction)
    def population_logits(self,logits,target_prior):
        """Under label shift: z approximately log p(x|y)+(1-tau)log q(y)."""
        p=torch.as_tensor(target_prior,device=logits.device,dtype=torch.float32)
        if p.shape!=(10,) or (p<=0).any(): raise ValueError('Invalid target prior')
        return logits.float()-(1-self.tau)*self.log_prior+(p/p.sum()).log()

```

### train_eval.py

```python
"""Train/evaluate Stage 2. No Stage-1 modifications; no implicit test-set selection."""
import argparse, csv, hashlib, json, math, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from dataset import CLASSES,NucleiDataset,read_manifest,sampling
from loss import LogitAdjustedCE
from model import Stage2,checkpoint_sha256

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def metrics(y, logits):
    y=np.asarray(y);logits=torch.as_tensor(logits,dtype=torch.float32)
    p=logits.softmax(-1).numpy();pred=p.argmax(-1)
    cm=np.bincount(10*y+pred,minlength=100).reshape(10,10)
    tp=cm.diagonal();support=cm.sum(1);predicted=cm.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(10,dtype=float),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(10,dtype=float),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(10,dtype=float),where=(support+predicted)>0)
    conf=p.max(1);correct=pred==y;ece=0.
    bins=np.minimum((conf*15).astype(int),14)
    for b in range(15):
        ix=bins==b
        if ix.any():ece+=ix.mean()*abs(conf[ix].mean()-correct[ix].mean())
    result={'macro_f1':float(f1.mean()),'balanced_accuracy':float(recall[support>0].mean()),
            'balanced_accuracy_fixed10':float(recall.mean()),'accuracy':float(correct.mean()),'ece15':float(ece),
            'nll':float(F.cross_entropy(logits,torch.as_tensor(y).long())),
            'support':support.tolist(),'predicted':predicted.tolist(),'precision':precision.tolist(),
            'recall':recall.tolist(),'f1':f1.tolist(),'confusion':cm.tolist(),
            'missing_classes':[CLASSES[i] for i in range(10) if support[i]==0]}
    return result

@torch.no_grad()
def evaluate(model,loader,device):
    model.eval();ys=[];zs=[];ids=[]
    for x,y,i in loader:
        z=model(x.to(device));ys.append(y);zs.append(z.float().cpu());ids.append(i)
    if not ys:raise ValueError('Empty evaluation split')
    y=torch.cat(ys);z=torch.cat(zs);ix=torch.cat(ids)
    return metrics(y.numpy(),z),y,z,ix

def train_epoch(model,loader,optimizer,criterion,device,accum=1,amp=False,scheduler=None):
    """Sum gradients then divide by actual window sample count, including remainder."""
    model.train();optimizer.zero_grad(set_to_none=True)
    total=0.;seen=0;window=0;exposure=torch.zeros(10,dtype=torch.long)
    use_amp=amp and device.type=='cuda'
    dtype=torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler=torch.amp.GradScaler('cuda',enabled=use_amp and dtype==torch.float16)
    trainable=[p for p in model.parameters() if p.requires_grad]
    positive_grad_sum=torch.zeros(10,dtype=torch.float64)
    captured={}
    head=getattr(model,'head',model if isinstance(model,nn.Linear) else None)
    handle=head.register_forward_pre_hook(lambda module,args:captured.update(h=args[0].detach())) if head is not None else None
    for step,(x,y,_) in enumerate(loader):
        x=x.to(device);y=y.to(device);n=len(y)
        with torch.autocast(device_type=device.type,dtype=dtype,enabled=use_amp):
            logits=model(x)
            loss=criterion(logits,y,reduction='sum')
        if 'h' in captured:
            with torch.no_grad():
                ptrue=(logits.float()+criterion.tau*criterion.log_prior).softmax(-1).gather(1,y[:,None]).squeeze(1)
                # Norm of each positive example's target-row head gradient before reduction.
                gn=(1-ptrue)*captured.pop('h').float().norm(dim=-1)
                positive_grad_sum.scatter_add_(0,y.cpu(),gn.double().cpu())
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss')
        scaler.scale(loss).backward();window+=n;total+=loss.item();seen+=n
        exposure+=torch.bincount(y.cpu(),minlength=10)
        if (step+1)%accum==0 or step+1==len(loader):
            scaler.unscale_(optimizer)
            for p in trainable:
                if p.grad is not None:p.grad.div_(window)
            torch.nn.utils.clip_grad_norm_(trainable,1.,error_if_nonfinite=True)
            old_scale=scaler.get_scale()
            scaler.step(optimizer);scaler.update()
            if scheduler is not None and scaler.get_scale()>=old_scale:scheduler.step()
            optimizer.zero_grad(set_to_none=True);window=0
    if handle is not None:handle.remove()
    model.positive_target_row_gradient_mean=(positive_grad_sum/exposure.clamp_min(1)).tolist()
    return total/seen,exposure.tolist()

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True)
    p.add_argument('--mode',choices=['linear','lora'],default='linear');p.add_argument('--sampler',choices=['natural','balanced'],default='natural')
    p.add_argument('--tau',type=float,default=1.);p.add_argument('--fov',type=int,default=96)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--patience',type=int,default=5)
    p.add_argument('--batch-size',type=int,default=8);p.add_argument('--accum',type=int,default=8)
    p.add_argument('--head-lr',type=float,default=1e-3);p.add_argument('--backbone-lr',type=float,default=1e-5)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--alpha',type=float,default=16.);p.add_argument('--last-blocks',type=int,default=4)
    p.add_argument('--workers',type=int,default=0);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--amp',action='store_true')
    p.add_argument('--checkpointing',action='store_true');p.add_argument('--stain-strength',type=float,default=0.)
    p.add_argument('--eval-checkpoint');p.add_argument('--eval-split',choices=['val','calibration','test'],default='val')
    a=p.parse_args()
    if a.accum<1 or a.epochs<1 or a.batch_size<1:raise ValueError('Positive batch/epoch/accum required')
    if a.sampler=='balanced' and a.tau!=0:raise ValueError('Use balanced sampler with tau=0; natural sampler with tau=0 or 1')
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);seed_all(a.seed);device=torch.device(a.device)
    rows=read_manifest(a.manifest)
    train=[r for r in rows if r['split']=='train'];val=[r for r in rows if r['split']=='val']
    sampler,q=sampling(train,a.sampler,a.seed)
    criterion=LogitAdjustedCE(q,a.tau).to(device)
    model=Stage2(a.weights,a.mode,a.rank,a.alpha,a.last_blocks,a.checkpointing).to(device)
    seed_all(a.seed)  # Augmentation stream must not depend on adapter initialization.
    def loader(rs,augment=False,sample=None):
        ds=NucleiDataset(rs,a.fov,augment=augment,stain_strength=a.stain_strength)
        return DataLoader(ds,batch_size=a.batch_size,shuffle=augment and sample is None,sampler=sample,num_workers=a.workers,pin_memory=device.type=='cuda',generator=torch.Generator().manual_seed(a.seed))
    config=vars(a).copy();config['classes']=CLASSES;config['sampling_prior']=q.tolist()
    config['manifest_sha256']=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest()
    config['weights_sha256']=checkpoint_sha256(a.weights)
    config['trainable_parameters']=sum(p.numel() for p in model.parameters() if p.requires_grad)
    if a.eval_checkpoint:
        ck=torch.load(a.eval_checkpoint,map_location=device,weights_only=True)
        if ck['config']['classes']!=CLASSES:raise ValueError('Checkpoint class map mismatch')
        if ck['config']['weights_sha256']!=config['weights_sha256']:raise ValueError('Wrong foundation checkpoint hash')
        for key in ['mode','rank','alpha','last_blocks','fov']:
            if ck['config'][key]!=config[key]:raise ValueError(f'Checkpoint configuration mismatch: {key}')
        current=model.state_dict();current.update(ck['trainable']);model.load_state_dict(current,strict=True)
        rs=[r for r in rows if r['split']==a.eval_split]
        m,y,z,ix=evaluate(model,loader(rs),device)
        (out/'evaluation.json').write_text(json.dumps(m,indent=2))
        np.savez_compressed(out/'predictions.npz',labels=y.numpy(),logits=z.numpy(),uids=np.array([rs[i]['uid'] for i in ix]))
        return
    if (out/'history.jsonl').exists():raise FileExistsError('Use a fresh output directory; never overwrite a research run')
    (out/'config.json').write_text(json.dumps(config,indent=2))
    groups=[]
    for role,lr in [('encoder',a.backbone_lr),('head',a.head_lr)]:
        for decay in [0.,.01]:
            params=[param for name,param in model.named_parameters() if param.requires_grad and name.startswith(role) and ((param.ndim>1)==(decay>0))]
            if params:groups.append(dict(params=params,lr=lr,weight_decay=decay))
    optimizer=torch.optim.AdamW(groups,betas=(.9,.999),eps=1e-8)
    tl=loader(train,True,sampler);vl=loader(val)
    total_steps=a.epochs*math.ceil(len(tl)/a.accum);warmup=max(1,round(.05*total_steps))
    def lr_factor(step):
        if step<warmup:return (step+1)/warmup
        progress=min(1.,(step-warmup)/max(1,total_steps-warmup))
        return .1+.9*.5*(1+math.cos(math.pi*progress))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_factor)
    best=-1.;stale=0
    for epoch in range(1,a.epochs+1):
        loss,exposure=train_epoch(model,tl,optimizer,criterion,device,a.accum,a.amp,scheduler)
        m,y,z,ix=evaluate(model,vl,device)
        rec=dict(epoch=epoch,objective=loss,class_exposure=exposure,validation=m,
                 positive_example_target_row_gradient=model.positive_target_row_gradient_mean)
        with (out/'history.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        print(json.dumps({'epoch':epoch,'loss':loss,'val_macro_f1':m['macro_f1'],'zero_recall':[CLASSES[k] for k in range(10) if m['recall'][k]==0]}),flush=True)
        if m['macro_f1']>best+1e-4:
            best=m['macro_f1'];stale=0
            names={name for name,param in model.named_parameters() if param.requires_grad}
            ck={'config':config,'epoch':epoch,'trainable':{k:v.detach().cpu() for k,v in model.state_dict().items() if k in names}}
            torch.save(ck,out/'best.pt')
            np.savez_compressed(out/'best_validation.npz',labels=y.numpy(),logits=z.numpy(),uids=np.array([val[i]['uid'] for i in ix]))
        else:stale+=1
        if stale>=a.patience:break

if __name__=='__main__':main()

```
