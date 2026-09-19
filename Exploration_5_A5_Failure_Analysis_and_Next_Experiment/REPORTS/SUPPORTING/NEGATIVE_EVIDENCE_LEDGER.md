# Negative-evidence lock

These are historical experiments, not Exploration5 reruns. P2 selected epochs differ from P3/P4 fixed10 endpoints. See the report for aggregate interpretation and scope.

|Exploration|Experiment|ROI-F1|Delta|Decision|
|---|---|---|---|---|
|3|A0_real|0.075209|-0.002892|not promoted|
|3|A1_real|0.078101|0.000000|not promoted|
|3|A2_placebo|0.075051|-0.003050|not promoted|
|3|A2_real|0.076840|-0.001261|not promoted|
|3|A2_shuffle|0.075524|-0.002577|not promoted|
|3|A3_placebo|0.075839|-0.002262|not promoted|
|3|A3_real|0.078337|0.000236|not promoted|
|3|A3_shuffle|0.075366|-0.002734|not promoted|
|3|A4_placebo|0.075839|-0.002262|not promoted|
|3|A4_real|0.078337|0.000236|not promoted|
|3|A4_shuffle|0.075366|-0.002734|not promoted|
|3|A5_placebo|0.072356|-0.005745|not promoted|
|3|A5_real|0.084098|0.005997|eligible for one confirmation|
|3|A5_shuffle|0.073396|-0.004704|not promoted|
|3|A6_placebo|0.074216|-0.003885|not promoted|
|3|A6_real|0.078534|0.000433|not promoted|
|3|A6_shuffle|0.075437|-0.002664|not promoted|
|3|B0_real|0.028881|-0.049220|diagnostic only|
|3|B1_real|0.077171|-0.000930|diagnostic only|
|3|B2_real|0.078834|0.000733|diagnostic only|
|3|B3_placebo|0.074649|-0.003452|diagnostic only|
|3|B3_real|0.081017|0.002916|diagnostic only|
|3|B3_shuffle|0.075485|-0.002616|diagnostic only|
|3|B4_placebo|0.077864|-0.000236|diagnostic only|
|3|B4_real|0.081939|0.003838|diagnostic only|
|3|B4_shuffle|0.077533|-0.000567|diagnostic only|
|4|A0|0.084098|0.000000|not promoted|
|4|A1|0.101731|0.017634|internal eligible; A3 final confirmation FAILED|
|4|A1_broken|0.053247|-0.030851|not promoted|
|4|A2|0.106654|0.022557|internal eligible; A3 final confirmation FAILED|
|4|A2_broken|0.069188|-0.014909|not promoted|
|4|A3|0.108758|0.024661|internal eligible; A3 final confirmation FAILED|
|4|A3_broken|0.058558|-0.025540|not promoted|
|4|A4|0.104686|0.020589|internal eligible; A3 final confirmation FAILED|
|4|A4_broken|0.086418|0.002320|not promoted|
|4|B1|0.084334|0.000236|not promoted|
|4|B2|0.085729|0.001631|not promoted|
|4|B_duplicate|0.084389|0.000292|not promoted|
|4|A0|0.084098|0.000000|not promoted|
|4|A1|0.101731|0.017634|internal eligible; A3 final confirmation FAILED|
|4|A1_broken|0.053247|-0.030851|not promoted|
|4|A2|0.106654|0.022557|internal eligible; A3 final confirmation FAILED|
|4|A2_broken|0.069188|-0.014909|not promoted|
|4|A3|0.108758|0.024661|internal eligible; A3 final confirmation FAILED|
|4|A3_broken|0.058558|-0.025540|not promoted|
|4|A4|0.104686|0.020589|internal eligible; A3 final confirmation FAILED|
|4|A4_broken|0.086418|0.002320|not promoted|
|4|B1|0.084334|0.000236|not promoted|
|4|B2|0.085729|0.001631|not promoted|
|4|B_duplicate|0.084389|0.000292|not promoted|
|4|D1|0.084035|-0.000063|not promoted|
|2|exp_biology_lr_0.003|0.050955|not recorded|historical selected endpoint, development only|
|2|exp_biology_lr_0.01|0.064772|not recorded|historical selected endpoint, development only|
|2|exp_biology_only|0.035750|not recorded|historical selected endpoint, development only|
|2|exp_fov128_la|0.118545|not recorded|historical selected endpoint, development only|
|2|exp_fov64_la|0.113452|not recorded|historical selected endpoint, development only|
|2|exp_fov96_la|0.125552|not recorded|historical selected endpoint, development only|
|2|exp_loss_balanced_ce|0.133167|not recorded|historical selected endpoint, development only|
|2|exp_loss_ce|0.121196|not recorded|historical selected endpoint, development only|
|2|exp_rep_none_29|0.116373|not recorded|historical selected endpoint, development only|
|2|exp_rep_none_43|0.113488|not recorded|historical selected endpoint, development only|
|2|exp_fusion_oracle|0.133667|not recorded|historical selected endpoint, development only|
|2|exp_fusion_placebo|0.133667|not recorded|historical selected endpoint, development only|
|2|exp_rep_placebo_29|0.117738|not recorded|historical selected endpoint, development only|
|2|exp_rep_placebo_43|0.117155|not recorded|historical selected endpoint, development only|
|2|exp_fusion_shuffle|0.132917|not recorded|historical selected endpoint, development only|
|2|exp_rep_shuffle_29|0.118123|not recorded|historical selected endpoint, development only|
|2|exp_rep_shuffle_43|0.115071|not recorded|historical selected endpoint, development only|
|2|exp_drop_gradient_texture|0.132833|not recorded|historical selected endpoint, development only|
|2|exp_drop_ring|0.134512|not recorded|historical selected endpoint, development only|
|2|exp_drop_roi_relative|0.136357|not recorded|historical selected endpoint, development only|
|2|exp_drop_stain|0.135774|not recorded|historical selected endpoint, development only|
|2|exp_family_gradient_texture|0.132095|not recorded|historical selected endpoint, development only|
|2|exp_family_ring|0.133167|not recorded|historical selected endpoint, development only|
|2|exp_family_roi_relative|0.133667|not recorded|historical selected endpoint, development only|
|2|exp_family_stain|0.130238|not recorded|historical selected endpoint, development only|
|2|exp_fusion_tierA|0.139607|not recorded|historical selected endpoint, development only|
|2|exp_rep_tierA_29|0.130929|not recorded|historical selected endpoint, development only|
|2|exp_rep_tierA_43|0.120851|not recorded|historical selected endpoint, development only|
|2|exp_tune_bio_lr_0.0003|0.118373|not recorded|historical selected endpoint, development only|
|2|exp_tune_bio_lr_0.003|0.128036|not recorded|historical selected endpoint, development only|
|4|LoRA|0.152605|0.000000|not promoted|
|4|frozen|0.152605|0.000000|not promoted|
|4|F_A3|0.227474|0.000000|not promoted|
|4|F_CLS_MASK|0.213479|-0.013995|not promoted|
|4|F_CLS_SHUFFLE|0.212460|-0.015014|not promoted|
|4|F_GAUSSIAN|0.239339|0.011865|not promoted|
|4|F_MASK|0.228203|0.000730|not promoted|
|4|F_SHUFFLE|0.215241|-0.012233|not promoted|
|4|C_ALIGNMENT|not recorded|0.005635|not promoted|