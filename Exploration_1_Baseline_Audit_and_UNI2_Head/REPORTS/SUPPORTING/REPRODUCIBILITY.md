# Reproduce the investigation

Extract `PUMA_STAGE2_RESEARCH_BUNDLE.zip` into a new directory. The `outputs` folder contains the implementation and report, while the `work` folder contains experiment scripts, fixed manifests, protocols, measurements and the small UNI2 feature cache.

Install `requirements.txt` in a separate Python environment. The measured environment was Python 3.10, PyTorch 2.11.0 CPU and timm 1.0.20. The experiment scripts use four CPU threads where specified. The original raw dataset and UNI2-h checkpoint paths remain in manifests/scripts; update paths explicitly if moving machines. Foundation weights and raw ROI images are not bundled.

From the extracted root:

```powershell
python test_integrity.py
python probe_uni2.py --seed 17
python probe_uni2.py --seed 29
python probe_uni2.py --seed 43
python discrimination.py
python analyze_results.py
python plot_results.py
python build_report.py
```

The existing JSON results allow report and figure regeneration without retraining. Rerunning probe scripts replaces their corresponding experimental outputs, so use a fresh extracted copy for each replication. `extract_uni2.py` refuses to overwrite an existing feature cache; re-extract only in a fresh workspace with the fixed manifest. The original feature cache records checkpoint SHA256 and crop settings in `uni2_features.json`. `BUNDLE_MANIFEST.json` records hashes of packaged files.

For CNN replication, `python cpu_controls.py` regenerates its pixel cache from the original ROI images when absent. For late-block LoRA replication, `python lora_dynamics.py` regenerates the frozen prefix cache when absent. Existing completed LoRA logs are skipped; preserve them and use a fresh results directory in a separate replication copy before rerunning. The preliminary `lora_results` logs are explicitly **excluded from model comparisons** because their minibatch shuffle streams differed. Valid paired results are in `lora_results_v2`.

The restricted LoRA probe uses constant learning rates and fresh common head initialization; it does not establish a full-data adapter benefit. Its trainable-weight files are diagnostic snapshots, not the deployment `best.pt` schema. Use the commands in `README.md` to produce inference-compatible checkpoints.

The archive excludes raw images, original documents, full foundation weights, installed dependencies, large pixel/prefix caches and temporary download URLs. The original Drive training history is included as source evidence, clearly separate from newly executed experiments. Protocol commit history is recorded in `protocol_git_log.txt`.

Scientific convergence is not established. Verified patient grouping, an untouched evaluation population and frozen Stage-1 proposals with provenance are required for the next validation stage. The installed runtime is CPU-only; no CUDA or full-data LoRA performance is claimed.
