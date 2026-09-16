# Measured CPU performance

Hardware and raw profiling details are in the adjacent JSON files. All encoder timings are eager FP32; cached-head timings exclude one-time encoder extraction. RSS is sampled process RSS, not an isolated tensor-memory allocation measurement.

| FOV | Examples | Wall seconds | CPU seconds | Examples/s | Peak sampled RSS GB |
| --- | --- | --- | --- | --- | --- |
| 96 | 450 | 743.62941 | 5708.59375 | 0.60514 | 5.78594 |
| 64 | 450 | 729.04698 | 5589.15625 | 0.61724 | 3.07156 |
| 128 | 450 | 661.78727 | 5176.81250 | 0.67998 | 2.96239 |


Selected CPU settings: eight threads, batch four, zero workers. All inputs resize to 224 square; timing differences across FOVs are not evidence of encoder complexity changes. Biology preprocessing: 30.1256 seconds for 450 nuclei across 181 ROIs. Loader two-worker startup outweighed its benefit. Eager cached-head execution beat the tested compile configuration, which also cost 64.26 seconds initially. No full-data or GPU runtime is claimed.

| Experiment | Head/evaluation wall seconds |
| --- | --- |
| exp_biology_lr_0.003 | 1.38379 |
| exp_biology_lr_0.01 | 1.30667 |
| exp_biology_only | 1.44177 |
| exp_drop_gradient_texture | 0.84641 |
| exp_drop_ring | 1.03566 |
| exp_drop_roi_relative | 1.06333 |
| exp_drop_stain | 1.23283 |
| exp_family_gradient_texture | 1.21535 |
| exp_family_ring | 1.21834 |
| exp_family_roi_relative | 1.18162 |
| exp_family_stain | 1.12544 |
| exp_fov128_la | 1.35439 |
| exp_fov64_la | 1.65097 |
| exp_fov96_la | 1.46576 |
| exp_fusion_oracle | 1.21721 |
| exp_fusion_placebo | 1.18072 |
| exp_fusion_shuffle | 1.21951 |
| exp_fusion_tierA | 1.14370 |
| exp_loss_balanced_ce | 1.23265 |
| exp_loss_ce | 1.55893 |
| exp_rep_none_29 | 1.20702 |
| exp_rep_none_43 | 1.06223 |
| exp_rep_placebo_29 | 1.12028 |
| exp_rep_placebo_43 | 1.05384 |
| exp_rep_shuffle_29 | 1.13537 |
| exp_rep_shuffle_43 | 1.08517 |
| exp_rep_tierA_29 | 1.19750 |
| exp_rep_tierA_43 | 1.12966 |
| exp_tune_bio_lr_0.0003 | 1.03060 |
| exp_tune_bio_lr_0.003 | 0.99899 |

