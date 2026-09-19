# Scope of the fixed-epoch scaling contrast

The executed design follows the historical fixed-final10-epoch control. Each epoch draws N samples, so D300/D600/D900 use5/10/15 optimizer updates per epoch at batch64. Total updates per seed are50/100/150. The last batches contain44/24/4 samples, respectively. These quantities are derived from the executed sampler/loader contract and verified logged draws; they were not additional logged optimizer-step counters.

The contrast changes both unique annotation availability and total update/exposure budget. It therefore measures the prescribed fixed-epoch scaling policy, not an isolated causal effect of diversity at identical updates. More annotation UIDs also do not prove more independent patients: all sizes use the same preassigned61-ROI TRAIN pool and patient mapping is unavailable. Do not claim that a threefold annotation increase means threefold independent groups. Existing per-epoch unique-ROI counts provide the observed exposure.

D900 improves the mean DEV/ROI metrics and reduces the semantic gap relative to D300, but it fails the neutrophil class-recall guard. No extra matched-budget run is introduced post hoc merely to rescue that outcome. C compares sampling at fixed D900 and matched joint-stage budgets; cRT has an explicitly matched five-extra-epoch no-reset control. Further study follows the original finite sequence.
