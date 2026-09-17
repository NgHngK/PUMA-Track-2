# Reflection after finite A/B tests

A5 reference: all nine seed/fold final logits exactly reproduced (maximum absolute error0). Token extraction450 complete with CLS/full-prefix parity.

A3 3x3 local pooling: meanROI-F1 .10875830237532368 vsA5 .08409771473601262; delta .02466058763931105,5000ROIbootstrap95% [.013820612405718788,.03609149499043116],all3seeds positive. Broken-coordinate control .058557919621749416. All promotion criteria pass. A2Gaussian .1066542834627941 is lower by .002104, outside .002parsimony band; all targetheads same27866params, chooseA3. A3still unconfirmed on original150.

B1(64+96) .08433412135539797;B2(96+128) .08572892040977148;sharedduplicate96control .0843892828999212. Neither passes .003gain or .002control thresholds. B3conditional not justified; no further multiscale addition. These reject tested rank8 zero-init residual form, not all possible multiscale methods.

Implementation incident: pooling geometry NPZ initially used duplicate key 'broken' for coordinates and weights. Extraction/encoder cache unaffected, no model ran before correction. Existing representation arrays verified exactly equal before resuming; no historical file altered. v1source preserved. B1/B2share a single identical duplicate96control, avoiding duplicate experiments. Reference runner scheduler-only edit while process ran did not affect loaded training logic; all exact reference logits prove reproduction. Frozen v1snapshot retained.

Next: proposal alignment provenance, D1ROI-class sampler on A5 and selectedA3, then restrictedLoRA. A3+D1 is only eligible if D1individually improves A5 and also improves A3. Existing originals unchanged.
