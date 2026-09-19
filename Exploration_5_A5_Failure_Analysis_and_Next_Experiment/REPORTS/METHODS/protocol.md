# Exploration 5 diagnostic protocol

Scope: retrospective, exploratory root-cause analysis only. No model promotion from reused data. Historical files are read-only, and Stage 1 is unchanged. The analysis focuses on identifying the failure, checking the evidence, and stopping once the next experiment is justified.

Inner loop 1: deduplicate historical run paths; extract every real A5 epoch and class trajectory; distinguish objective from population CE, and semantic from fixed-ten ROI metrics. Missing original measurements are explicitly NOT RECORDED IN ORIGINAL RUN.

Inner loop 2: verify cache hashes and UID order, then calculate cosine class/group geometry, held-fold 5NN purity, centroid shifts and covariance distances for CLS/target/Gaussian/neighborhood. No hyperparameters fitted to outcomes. Cosine normalize each representation, use training-only centroids, and exclude entire held ROIs. Summaries are descriptive, not causal tests. Tier-A associations are exploratory; group eta-squared has cardinality bias and cannot alone prove confounding.

Inner loop 3: compare preserved fixed-epoch confirmation confusion matrices and class supports. Do not retrain, select an epoch, or rescue a damaged class.

Outer loop: rank H1–H10 using supporting and contradictory evidence. Only then search targeted primary literature. At most three candidates; one recommended next experiment. If no clean full-population A5 baseline exists, prepare its protocol and admission gates before considering architecture expansion.

Future promotion rule (locked before any new training): paired ROI-F1 gain >=0.003, >=2/3 positive seeds, group-bootstrap 95% lower bound >0, mean class recall loss <=0.10, gap increase <=0.05, matched control advantage >=0.002 where applicable, clean provenance, and one locked confirmation. No candidate training is authorized by a favorable retrospective best epoch.
