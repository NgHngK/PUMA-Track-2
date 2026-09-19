# Exploration5 research artifacts

Start with [EXPLORATION5_ROOT_CAUSE_REPORT.md](../MAIN/EXPLORATION5_ROOT_CAUSE_REPORT.md). The result is **retain A5; no new architecture justified**. The next full-population baseline is prepared, not executed, because upstream outer-exclusion and locked evaluation admission are unresolved.

All new work is in this directory; historical Exploration1–4 files remain unchanged. Diagnostics are retrospective/exploratory. `VERIFICATION.json` records cache UID alignment and consumed-source immutability.

Reproduce with Python, numpy and pandas: `src/analyze.py`, then `src/supplement.py`, then `src/verify_and_localize.py`, then `src/finalize.py`. The scripts read the archive sibling directory and the recorded local raw GeoJSON path. `src/figures.py` adds matplotlib and saves PNG/PDF figures. `src/analyze.py` regenerates the preliminary ledger; always run supplement afterwards to restore the complete90-row ledger. No training occurs. Paths and exact input hashes are retained in SOURCE_HASHES.json. The local bundled Python path and package versions are in ENVIRONMENT.json.

The pre-analysis exploratory protocol is committed as4461dee. No retrospective diagnostic metric is represented as a preregistered confirmatory result. Full-data training requires a new, admitted and locked UID split before execution; see FULL_DATA_A5_PROTOCOL.md and baseline_config.json.
