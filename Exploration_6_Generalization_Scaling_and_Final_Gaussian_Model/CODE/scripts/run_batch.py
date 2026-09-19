#!/usr/bin/env python
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path

def main() -> None:
    p=argparse.ArgumentParser();p.add_argument('--config-dir',required=True);p.add_argument('--pattern',default='*.json');p.add_argument('--stop-on-error',action='store_true');a=p.parse_args()
    root=Path(a.config_dir)
    if not root.is_dir():raise FileNotFoundError(root)
    batch_record=root/'batch_results.json'
    if batch_record.exists():raise FileExistsError(f'batch_results.json already exists; preserve it and use a new batch directory or remove only after explicit archival: {batch_record}')
    metadata_names={'HPO_STAGE.json','SEED_BATCH.json','batch_results.json'};configs=[x for x in sorted(root.glob(a.pattern)) if x.is_file() and x.name not in metadata_names]
    if not configs:raise ValueError(f'no experiment configs matched {a.pattern!r} in {root}')
    results=[]
    for cfg in configs:
        try:
            raw=json.loads(cfg.read_text(encoding='utf-8'))
            if not isinstance(raw,dict) or not raw.get('experiment_id'):raise ValueError('missing experiment_id')
        except Exception as e:
            rec={'config':str(cfg),'returncode':2,'last_stdout':'','stderr_tail':f'invalid experiment config: {e}'};results.append(rec);print(json.dumps(rec,separators=(',',':')),flush=True)
            if a.stop_on_error:break
            continue
        proc=subprocess.run([sys.executable,str(Path(__file__).with_name('run_experiment.py')),'--config',str(cfg)],text=True,capture_output=True)
        rec={'config':str(cfg),'returncode':proc.returncode,'last_stdout':proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else '', 'stderr_tail':'\n'.join(proc.stderr.strip().splitlines()[-10:])};results.append(rec);print(json.dumps(rec,separators=(',',':')),flush=True)
        if proc.returncode and a.stop_on_error:break
    batch_record.write_text(json.dumps(results,indent=2),encoding='utf-8')
    if any(r['returncode'] for r in results):raise SystemExit(1)

if __name__=='__main__':
    main()
