# Exploration-6 config registry

These are **templates**, not a command to execute every file. Replace `PATH/TO/...` before use.

- `00_*`: data-scaling/historical A5 controls. `D300`, `D600`, and `D900` must each use a cache and Tier-A sidecar bound to that exact manifest; rank-8 A5, inverse replacement + CE, constant LR, fixed epoch 10.
- `10-13_*`: raw-image augmentation arms. `13` is only valid after empirical Stage-1 displacement confirms the chosen sigma/clip.
- `20-21_*`: long-tail and cRT-style decoupled classifier arms.
- `30-31_*`: regularization arms.
- `40_*`: Tier-A ablation.
- `50_*`: Exploration-4 representation replication, including exact fixed A4-style global-local pooling.
- `90_*`: conditional encoder adaptation. **Do not run unless Exploration-6 adaptation gate opens.** It is intentionally not the exact failed Exploration-4 Q/V r8/alpha16/last4/5-epoch configuration.

HPO should be generated from a stabilized base config with `scripts/generate_hpo.py`, not by running every template blindly.

For final three-seed replication, use `scripts/make_seed_replicates.py` on a frozen finalist config. Do not hand-edit the three copies independently.
