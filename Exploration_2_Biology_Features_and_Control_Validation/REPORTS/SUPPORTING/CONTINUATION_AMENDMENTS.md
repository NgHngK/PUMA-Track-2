# Prospectively recorded amendments and implementation findings

1. The local V17 annotation loader expands MultiPolygon exterior rings. The initial join incorrectly treated the expanded row index as an original GeoJSON feature index and failed a class assertion before training. The corrected join uses original feature UID, verifies its class name, and records every V17 component centroid. No failed join output entered an experiment. Full semantic census97,193; component census97,378;163 MultiPolygon annotations. The expanded sample retains450 semantic features.

2. The previous FOV96 cache lacks image-content hashes and uses a different manifest. It remains archived unchanged, but is not imported into the stricter new cache. All450 nuclei are recomputed at each FOV. The new cache keys contain manifest, image, checkpoint and transform/source hashes and refuse mismatched resume. Cached outputs are reused across all subsequent head/fusion/loss runs.

3. Measured eight-thread batch4 CPU throughput exceeded four-thread alternatives by about2×. Select this setting. CPU backend reports AVX2, so no presumed hardware BF16 speedup is used. Compilation is not promoted without a suitable measured amortization case for this finite extraction run.

4. ROI path resolution originally repeated per nucleus. Shared manifest reading now resolves and checks each unique image path once while preserving leakage checks, avoiding repeated filesystem work. Input semantics are unchanged.

5. Biology-only LR1e-3 ends epoch10 at training Macro-F1≈0.114 and objective2.206, with ongoing improvement and similar validation performance. This is under-optimization evidence, not a test of maximal biology separability. Before declaring biology uninformative, run the same linear model at LR3e-3 and1e-2 for ten epochs; no MLP or extra descriptor is introduced. Label these post-finding optimization diagnostics exploratory. They do not select a different fusion architecture by themselves.

6. The original protocol's five-epoch successive-halving LR refinement is reserved for the eventual survivor. Implementation must retain optimizer and RNG state if extending a partial run; otherwise use explicitly new complete ten-epoch runs and record that amendment before execution. No fake continuation or best-of-many seed report is permitted.
