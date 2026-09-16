# Prospective protocol 2026-09-07

Scope: fixed Stage 1; ten-class semantic classification only. No tissue features, biological fusion, or independent debate agents.

H1: Natural-sampling CE can lower training loss while neglecting tail classes. Compare natural CE, natural logit-adjusted CE (tau=1), and inverse-frequency sampling with ordinary CE, using the same small image CNN and real GT-centred crops. This is a mechanism screen, not a UNI2 experiment. Ten epochs, fixed seed 17, group-disjoint ROI split. Include all classes in train and validation and report exact supports. Use up to 100 nuclei per ROI sampled uniformly, retaining every ROI; evaluate natural subsample distribution. Missing patient identifiers make this ROI-separated, not verified patient-separated. Primary diagnostic: macro10 F1 and per-class recall; also unaugmented training CE, validation CE, balanced accuracy, ECE, class exposure and positive-example gradient norms. No requirement for strictly monotonic minibatch loss; report violations without rerunning to hide them.

H2: Training-time logit adjustment CE(z+log(pi),y) with natural sampling differs from post-hoc adjustment CE(z,y) then z-log(pi). Check analytical gradients, correct adjustment sign, equivalence to Balanced Softmax at tau=1, and sampler-implied priors. The original ratio text is used only for illustrative gradient/batch-absence calculations; actual training uses fold counts.

H3: A frozen local ViT-L/16 checkpoint may provide usable real pretrained features. First verify strict loading and CPU forward runtime. It is not UNI2-h; mark provenance-unverified and do not promote a UNI2 architecture from its scores. Run a bounded representative frozen-feature screen if feasible. Actual UNI2/LoRA remains unvalidated until correct weights are available.

Evaluation is a development screen; no untouched test result or SOTA claim. Epoch and recipe selection are exploratory and require subsequent independent grouped confirmation. Do not compare to reported leaderboard values.

Implementation tests: crop centring at boundaries; disjoint split/UID checks; class-map schema; wrong-checkpoint rejection before allocation; nonzero LoRA B gradients and delayed A gradient under zero-B initialization; gradient accumulation partial-window equivalence; absent-class metrics; finite loss and probabilities; checkpoint reload predictions.
