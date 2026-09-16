# Exact supplied local V17 contract

The evaluator source is vendored with the package. Byte-identical source files and their SHA256 values are recorded in `source_hashes.json`. Regression results are in `regression_test.json`.

The local implementation describes itself as a behavioral clone of public Track2. This investigation establishes parity with that supplied V17 implementation; it does not independently certify equivalence to any later live challenge server.

## Class mapping

Research IDs remain exactly tumor0, lymphocyte1, plasma2, histiocyte3, melanophage4, neutrophil5, stroma6, epithelium7, endothelium8, apoptosis9. Local V17 constants instead have endothelium7 and epithelium8. The boundary maps research IDs to V17 IDs `[0,1,2,3,4,5,6,8,7,9]` through full class names. Vendored constants are not edited. Output metric dictionaries use explicit names.

## GT and prediction geometry

`annotations.py::load_nuclei_polygons` emits every exterior polygon ring, expanding MultiPolygon annotations. Interior holes do not contribute a separate GT. `annotation_centroid` casts serialized coordinates to float32, computes mean over every path point in float64, and casts the result to float32. A repeated closing point contributes again. `public_matcher.py::calculate_public_centroid` means provided path points in float64; it does not compute an area centroid.

Prior semantic crops use Shapely area centroids and one row per annotation feature. New controls preserve these centers for continuity; fresh caches were extracted because the legacy cache lacked image-content hashes. Their GT evaluation includes each corresponding exterior component at the V17 center. Therefore V17 GT support can exceed semantic annotation-feature support. These are subset GT-centred controls, not full-ROI detection results. Omitted nonsampled GT/predictions are not evaluated.

`dataset_evaluation.py` formats feature dictionaries with uid, category, centroid, score. `prediction_features` uses input score, default1. The historical trainer computes semantic confidence times utility for its own predictions. The new conditional controls have no utility branch: score is maximum semantic probability, recorded as part of the control contract. Stage1 coordinates, scores, thresholds and suppression remain untouched. This difference in score production is explicit; matcher behavior is identical.

## Matching

`PublicMatcher.match` processes GT in supplied order. Predictions are partitioned by category. Eligibility requires same category and Euclidean distance **strictly less than15 pixels**. Eligible predictions are sorted by descending score, then ascending distance, then stable input order on a complete tie. Select the first.

Deletion intentionally finds the first same-category remaining prediction with an exactly equal centroid, even if its UID differs from the selected UID. Duplicate-centroid collisions can therefore select the same UID more than once. This behavior is preserved and tested; it is not repaired into Hungarian or unique-UID matching. Traces record GT UID/category, selected/removed UIDs, repeated selection, collision flag, score, distance and eligible UIDs.

## Counts and aggregation

For each category: TP is the number of matching events; FP=number of original predictions−TP; FN=number of GT features−TP. Precision=TP/(TP+FP), recall=TP/(TP+FN), and F1=2PR/(P+R), with zero for zero denominators. Local per-ROI metrics contain union-of-present GT/predicted categories, micro P/R/F1 and macro F1 over that union.

`aggregate_fixed10` fills missing class metrics with zero, averages each class P/R/F1 across **all requested ROIs**, then averages the ten class averages. Rare classes absent from an ROI contribute zero. `aggregate_public_dynamic` uses the category union across requested ROIs and otherwise the same averaging. `aggregate_summed` sums TP/FP/FN across ROIs for each of ten classes, recomputes class P/R/F1, then macro-averages classes. No requested ROIs yields NaN macro values. Duplicate requested ROI indices raise an error. ROI fixed10 and summed/pooled scores are distinct estimands.

The returned schema is evaluation_contract_id, public_dynamic, fixed10, summed, roi_metrics, traces. Contract ID is `puma_public_track2_nuclei_class_first_greedy_dynamic_avg_2026`.

## Accuracy and supplemental metrics

The V17 matcher/aggregators define no official accuracy. New controls report **conditional semantic accuracy**: correctly classified sampled annotation features / all sampled annotation features. Each feature contributes once; V17 matched/unmatched component decisions do not alter this denominator. For a future fixed-Stage1 matched classification cohort the same metric is conditional on an explicitly documented association; unmatched GT and unmatched proposals are excluded only from this supplemental semantic denominator and remain in end-to-end PUMA counts. Do not call it official PUMA accuracy.

Balanced accuracy, semantic confusion, NLL, ECE15, train/validation gaps and prevalence are supplemental. On enriched subsets, NLL/ECE are development calibration diagnostics. No clinical calibration claim follows.

## Serialization

`serializer.py` orders predictions by V17 class ID, descending score, x, y and UID. It emits four symmetric vertices without repeating the closing vertex, so their arithmetic mean preserves the center; score is consumed by the matcher. A probability field is also emitted for tooling compatibility. The new experiment evaluator calls V17 dataset evaluation directly with an explicitly fixed source-row order. Final exported submissions must use the original serializer and canonical-name remapping; serialization order is a documented tie-break difference from arbitrary arrays.

## Parity coverage

Six deterministic fixtures cover score-before-distance, strict-radius boundary, duplicate centroid deletion, stable tie, wrong class and empty inputs. One hundred seeded random fixtures compare complete matches, traces and metrics. A two-ROI fixture compares all aggregation outputs and absent-class behavior. A boundary-label fixture verifies epithelium/endothelium mapping. These tests establish the recovered behavior before architecture comparisons.
