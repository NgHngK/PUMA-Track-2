# Exploration-6 architecture / training matrix

| Family | ID / config | What changes | Prior evidence status | Default eligibility |
|---|---|---|---|---|
| Historical control | `00_historical_A5_cached` | Nothing: CLS + Tier-A A5 rank8 | retained Exploration-3 baseline | required control |
| Augmentation | AUG0 | raw-image parity, no augmentation | previously unvalidated in A5 | required before AUG1 |
| Augmentation | AUG1 | flips + 90° rotations | proposed, not validated | open |
| Augmentation | AUG2 | AUG1 + fixed H/E OD concentration perturbation | proposed, not validated | open |
| Coordinate robustness | AUG3 | empirical Stage-1 center jitter | Exploration-4 shift measured | gated by measured displacement |
| Long-tail | LT0 | inverse replacement + CE | historical selected recipe | required control |
| Long-tail | LT1 | natural / alpha=.5 tempered sampling | partially studied family | open |
| Long-tail | LT2 | cRT-style reset of W/R with Ph/Pb frozen | true decoupled hypothesis | open if justified |
| Regularization | rank2/4/8 | A5 interaction capacity | not systematically tested | open |
| Regularization | dropout / smoothing / WD / schedule | generalization | not systematically tested | open |
| Tier-A | T0/T1/T2 | off/on/regularized | useful but confounding concern | open |
| Representation | REP0 | CLS | retained baseline | required control |
| Representation | REP1 | Exploration-4 local Gaussian/3x3 | positive internal signal, A3 confirmation failed | replication only |
| Representation | REP2 | exact Exploration-4 A4 global-local formula | positive unconfirmed | fresh replication |
| Representation | scalar mix | one learned scalar | not previously established | only after fixed REP2 passes |
| Encoder | conditional LoRA | materially different PEFT candidate | exact old LoRA failed | only if adaptation gate opens |

## Exact retained A5

For non-affine-LN representation `h` and train-standardized Tier-A `b`:

`z = W h + a + R((P_h h) ⊙ (P_b b))`

Baseline dimensions: `W:1536→10`, `P_h:1536→8`, `P_b:16→8`, `R:8→10`, with `R=0` at initialization; **27,866** trainable parameters.

## Exact Exploration-4 A4 replication

`h = LN((LN(CLS) + LN(GaussianLocal))/2)`

Gaussian spatial-token weights use sigma `1.5` patch units over the 16×16 token grid. No learned pooling parameters are added.
