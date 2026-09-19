from pathlib import Path
import json
p=Path('PUMA_STAGE2_COLAB');cells=[]
def md(s):cells.append(dict(cell_type='markdown',metadata={},source=s.splitlines(True)))
def code(s):cells.append(dict(cell_type='code',metadata={},execution_count=None,outputs=[],source=s.splitlines(True)))
md('''# PUMA Exploration6 — finite GPU augmentation continuation
Upload the entire **PUMA_STAGE2_COLAB** folder to `/content/drive/MyDrive/Research/PUMA/`. Open this notebook in Colab, select **Runtime → Change runtime type → GPU**, then **Run all**.

This executes the remaining **seven Group-B trials**, preserving completed controls and the archived interrupted seed43 attempt. It does not claim to automate unresolved scientific decisions in D–J. Both holdouts remain unopened. No Codex automation or paid service is started.

The unchanged certified trainer has no exact interrupted-epoch resume. Completed trials are skipped; partial trials stop safely. Do not delete partial outputs or run two notebooks against this folder. Colab session availability may limit completion; return any stopped run for recovery review.''')
code('''from pathlib import Path
from google.colab import drive
drive.mount('/content/drive')
PROJECT_ROOT = Path('/content/drive/MyDrive/Research/PUMA')
PACKAGE = PROJECT_ROOT / 'PUMA_STAGE2_COLAB'
assert (PACKAGE / 'scripts/colab_runner.py').is_file(), f'Upload the entire folder to {PACKAGE}'
''')
code('''import subprocess, sys, os
# Preserve Colab's preinstalled CUDA-compatible torch/torchvision pair.
requirements = [line for line in (PACKAGE/'CODE/requirements.txt').read_text().splitlines() if line and not line.startswith('torch')]
subprocess.run([sys.executable, '-m', 'pip', 'install', *requirements], check=True)
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime, then rerun.'
print('GPU:', torch.cuda.get_device_name(0), 'VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory/2**30,1))
print('PyTorch:', torch.__version__)
''')
md('''## Validate and stage inputs
This verifies every uploaded file and copies runtime inputs to local Colab storage for faster reads. Original Drive inputs stay immutable. Only execution-view paths and their cache identity metadata are relocated. Outputs go directly to Drive.

CPU and CUDA can differ numerically. The physical batch remains 1, accumulation 64, AMP off; no unregistered batch-size or precision changes. CPU AUG0 seeds17/29 remain historical controls; device provenance is retained for interpretation.''')
code('''subprocess.run([sys.executable, str(PACKAGE/'scripts/colab_runner.py'), 'prepare', '--package', str(PACKAGE)], check=True)
''')
md('''## Run the finite queue
Leave this cell running. Results, epoch histories, per-class metrics, confusion matrices, checkpoint files, configs, and logs persist in `COLAB_RESULTS` on Drive. The trainer logs full epoch records rather than printing a progress bar. Check `COLAB_RESULTS/EXPERIMENTS/<run>/history.jsonl` for progress.

A dependable GPU runtime estimate requires the first measured run. There are seven 10-epoch jobs. After the first completion, multiply its duration by six for an initial remaining-time estimate; geometry/stain costs and Colab GPU type can change this. CPU measurements were 8.74–9.49 hours per job; they are not GPU estimates.''')
code('''subprocess.run([sys.executable, str(PACKAGE/'scripts/colab_runner.py'), 'run', '--package', str(PACKAGE)], check=True)
''')
code('''import json
print((PACKAGE/'COLAB_RESULTS/STATUS.json').read_text())
print('Keep the entire COLAB_RESULTS folder. Reopen the chat with its status and results for the next gated stage.')
''')
md('''## Next review
The notebook summarizes three-seed augmentation comparisons, ROI bootstrap intervals, class-harm guards, and raw/cached parity. Conditional C resolution, bounded D HPO, E/F and gated G, final seed/holdout stages, and the master report remain pending. Their frozen instructions, configs, prior results, and lineage are in `CONTEXT`. No holdout prediction or master-report append occurs here.''')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(colab=dict(name='PUMA_STAGE2_MAIN.ipynb'),kernelspec=dict(display_name='Python 3',language='python',name='python3'),accelerator='GPU'),cells=cells)
(p/'PUMA_STAGE2_MAIN.ipynb').write_text(json.dumps(nb,indent=2))
(p/'README.md').write_text('''# PUMA Stage2 Colab upload package

Upload this entire folder as:
`MyDrive/Research/PUMA/PUMA_STAGE2_COLAB/`

Open `PUMA_STAGE2_MAIN.ipynb` in Google Colab, select a GPU runtime, and Run all. The notebook calls `scripts/colab_runner.py` from Drive. No paths need editing if this layout is used.

- INPUTS: all 205 ROI TIFFs and nuclei GeoJSONs, frozen selected manifests, caches, exact UNI2 weights.
- CODE: executed certified code snapshot and tests, unchanged.
- CONTEXT: original resolved configs, completed study results, archived interrupted attempt, instructions, reports, tables, provenance and internal folds. These retain original Windows paths as historical evidence.
- COLAB_RESULTS: created on Drive by the notebook; all new outputs and resolved Linux/CUDA configs.
- PACKAGE_SHA256.json: immutable upload inventory; checked before execution.

Seven remaining augmentation trials are queued. Completed CPU AUG0 seeds17/29 are reused, never rerun. The authorized seed43 recovery starts from initialization because saved checkpoints lack optimizer/scheduler/RNG resume state; its six-epoch interrupted attempt is excluded and preserved in CONTEXT/INTERRUPTED_ATTEMPTS. Physical batch1, accumulation64 and FP32 stay fixed. CUDA is a documented execution-environment change, not a claim of bitwise equivalence. Interpret mixed-device control comparisons with this limitation.

The package stops at the mandatory B review gate. Later A–J study stages and integrated master report are not yet complete and are not silently guessed by the notebook. Both final holdouts remain unopened. Return COLAB_RESULTS to this chat for the next finite stage.

Do not run multiple copies simultaneously. An existing partial run causes a safe stop and requires reviewed recovery; there is no automatic restart. A disconnected browser does not guarantee the Colab backend remains alive. Results already written to Drive are retained.

GPU timing is measured by the first completed trial; no GPU hardware was available for local validation. Notebook syntax, portable configuration logic, package hashes and copied source are checked locally. Initial verification/staging requires roughly twice the package size in available local runtime storage.
''')
print('Notebook and instructions created')
