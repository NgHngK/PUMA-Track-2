# Exploration-6 final QA certificate

Final certification was performed after all protocol fixes.

## Five consecutive clean rounds

Each clean round passed all of the following without any code change between rounds:

- full automated test suite: **74/74 passed**;
- Python compileall over `src`, `scripts`, `tests`, and `vendor`;
- strict parsing/validation of all **22 shipped experiment JSON configs**;
- scan for development-container absolute paths and unfinished TODO/FIXME/NotImplemented markers;
- byte-for-byte Exploration-3 A5 reference parity against the supplied A5 package;
- byte-for-byte vendored PUMA V17 evaluator parity against the supplied A5 package;
- `--help` execution for every shipped CLI script;
- wheel build with no dependency download;
- clean wheel installation into an isolated target directory;
- import of the installed package, ten-class ontology, and final inference module.

The full pytest suite includes numerical A5 forward parity, parameter counts, token pooling, augmentation coordinate transforms, Tier-A, sampling/loss contracts, x3 nested split quotas, grouped TRAIN-only CV-HPO, cache hash/row-identity integrity, Windows-safe memmap pickling, cached/raw synthetic data flow, exact gradient accumulation, cRT/decoupled training, deterministic replay, locked evaluation, unlabeled proposal inference, and exact V17 end-to-end evaluation.

## Final x3 definition

The dataset builder preserves the actual historical Exploration-3 cohort class-support lineage:

- D300 TRAIN: tumor 71, lymphocyte 37, each remaining class 24;
- D600 TRAIN: tumor 142, lymphocyte 74, each remaining class 48;
- D900 TRAIN: tumor 213, lymphocyte 111, each remaining class 72;
- fresh DEV225: tumor 58, lymphocyte 23, each remaining class 18;
- fresh LOCKED225: tumor 59, lymphocyte 22, each remaining class 18.

The builder refuses an already-partitioned source pool and prioritizes the strongest supplied independent group identifier.
