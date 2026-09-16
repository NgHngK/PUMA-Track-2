import json,csv,hashlib,zipfile,shutil,re,sys
from pathlib import Path
W=Path(__file__).resolve().parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';D=R/'93_final_architecture'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
(D/'parameter_table.csv').write_text('component,total_parameters,trainable_parameters\nUNI2_h_checkpoint_tensors,681394176,0\nappearance_linear,15370,15370\nbiology_linear,160,160\ncomplete_model,681409706,15530\n',encoding='utf-8')
(D/'ENVIRONMENT.json').write_text(json.dumps({'tested_python':sys.version,'torch':'2.11.0+cpu','timm':'1.0.20','numpy':'2.2.6','device':'CPU only verified','encoder_checkpoint_tensor_count':681394176,'frozen_encoder':True,'head_parameters':15530,'final_source_is_separate_from_immutable_experiment_core':True},indent=2))
(R/'README.md').write_text('''# PUMA Stage 2 continuation — completed bounded study

Start with [STAGE2_RESEARCH_REPORT.md](STAGE2_RESEARCH_REPORT.md). It integrates 30 new ten-epoch CPU probes, the historical record, evaluator parity, all class metrics and exactly one next full-scale architecture.

The chosen architecture is frozen UNI2-h / FOV96 / balanced CE / a 16-feature zero-initialized linear biology correction. These are GT-centered development controls; full Stage 1 population performance and SOTA are not established. Stage 1 was unchanged.

* `00_reference`: copied user prompt, census, historical indexes, literature, parity and protocols.
* `01_sample_definition`: exact 450-feature manifest and QC.
* `01_shared_core`: immutable code used for all 30 runs (one revision, 480 per-module provenance checks).
* `02_cache`: three complete FP32 UNI2 caches and biology/oracle caches; no raw checkpoint or images.
* `experiments`: thin configurations, raw logs, selected checkpoints and predictions; no per-experiment code duplication.
* `90_comparative_analysis`: all metrics, paired promotion statistics and redundancy diagnostic.
* `91_error_analysis`: disagreement records and inspected contact sheet.
* `92_figures`: PNG and vector PDF figures.
* `93_final_architecture`: standalone documented code for the one next full-data experiment.

The standalone extractor uses a stricter cache interface than the research screen. Do not rename experimental caches into final-package caches: regenerate once for a verified full Stage 1 manifest. Research checkpoints are for reproduction, not deployment. The final model head was checked against saved experiment predictions (maximum absolute logit difference 1.19e-7).

The reproduction scripts under `00_reference/reproduction_scripts` preserve the original workspace orchestration and relative paths; they are archival source, not new entrypoints from that directory. Replaying the original screen uses `python 01_shared_core/train.py --config experiments/<run>/config.json` from this research root with unchanged external image paths, installed dependencies and a new output ID. Completed run directories are deliberately not overwritten. The final architecture README contains standalone full-data commands.

ARTIFACT_SHA256.csv checks every bundled artifact except itself. ZIP integrity is separately recorded beside the archive. Raw dataset images/annotations and pretrained weights remain at their original external paths and are not bundled. The copied reports are historical evidence, not current instructions.
''',encoding='utf-8')
check={};count=0
for p in (R/'experiments').glob('*/provenance.json'):
    s=json.loads(p.read_text())
    for n,h in s['modules'].items():assert sha(R/'01_shared_core'/n)==h;count+=1
check['immutable_core_module_hashes_verified']=count
text=(R/'STAGE2_RESEARCH_REPORT.md').read_text(encoding='utf-8')
links=re.findall(r'\]\(([^)]+)\)',text)
bad=[l for l in links if ':' not in l and not l.startswith('#') and not (R/l).exists()];assert not bad,bad
check['report_relative_links']='PASS';check['new_runs']=len(list((R/'experiments').glob('*/metrics/summary.json')))
check['report_characters']=len(text);check['figure_inspection']='Five figure types generated; paired controls, learning and per-class manually viewed; FOV/loss inspected during packaging'
check['standalone']=json.loads((D/'VERIFICATION.json').read_text())
(R/'00_reference/FINAL_AUDIT.json').write_text(json.dumps(check,indent=2))
state=json.loads((R/'00_reference/RESEARCH_STATE.json').read_text());state['final_verification']=check['standalone'];state['artifact_audit']='FINAL_AUDIT.json';(R/'00_reference/RESEARCH_STATE.json').write_text(json.dumps(state,indent=2))
for name in ['research-log.md','findings.md']:
    p=W/name;t=p.read_text(encoding='utf-8')
    marker='2026-09-08 continuation complete:'
    if marker not in t:p.write_text(t+'\n\n'+marker+' 30 additional ten-epoch controls finished after exact local V17 parity. One immutable shared core passed 480 module-hash checks. Selected frozen UNI2 FOV96 + balanced CE + 16-feature linear TierA correction (15,530 trainable parameters). Three-seed mean ROI-F1 delta +0.009453, exploratory ROI bootstrap [0.001355,0.017505]. Integrated report and standalone package saved under outputs/STAGE2_RESEARCH_20260907. GT-centered subset only; full Stage1 outer-group provenance and external validation remain next-study requirements. No Stage1 changes.\n',encoding='utf-8')
p=W/'research-state.yaml';t=p.read_text();t=t.replace('phase: report_and_reproducibility_package','phase: continuation_complete_full_scale_not_executed')
if 'continuation_new_runs:' not in t:t+='\ncontinuation_new_runs: 30\ncontinuation_report: outputs/STAGE2_RESEARCH_20260907/STAGE2_RESEARCH_REPORT.md\ncontinuation_architecture: frozen_UNI2_FOV96_balanced_CE_TierA_linear16\n'
p.write_text(t)
def files():return sorted(p for p in R.rglob('*') if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts and p.suffix!='.pyc' and p.name!='ARTIFACT_SHA256.csv')
with (R/'ARTIFACT_SHA256.csv').open('w',newline='') as f:
    wr=csv.writer(f);wr.writerow(['path','bytes','sha256']);wr.writerows((p.relative_to(R).as_posix(),p.stat().st_size,sha(p)) for p in files())
dest=W.parent/'outputs/PUMA_STAGE2_CONTINUATION_BUNDLE.zip'
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in files()+[R/'ARTIFACT_SHA256.csv']:z.write(p,Path(R.name)/p.relative_to(R))
with zipfile.ZipFile(dest) as z:assert z.testzip() is None;members=len(z.namelist())
out={'archive':dest.name,'sha256':sha(dest),'bytes':dest.stat().st_size,'members':members,'CRC_test':'PASS','new_runs':30,'full_data_validation':False}
dest.with_suffix('.verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
