from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from puma_exploration6.hpo import write_hpo_configs


def _base_config(tmp_path: Path) -> dict:
    return {
        "experiment_id":"base",
        "manifest":str(tmp_path/"manifest.csv"),
        "output_dir":str(tmp_path/"runs"),
        "cached_features":str(tmp_path/"features.npy"),
        "tier_a":str(tmp_path/"tier.npy"),
        "device":"cpu",
        "model":{"representation":"cls","use_tier_a":True},
        "sampler":{"mode":"inverse","alpha":1.0},
        "loss":{"kind":"ce","tau":0.0},
    }


def test_hpo_output_directory_is_immutable(tmp_path: Path):
    base=tmp_path/"base.json";base.write_text(json.dumps(_base_config(tmp_path)))
    out=tmp_path/"hpo"
    write_hpo_configs(base,out,2,"optimizer",17)
    with pytest.raises(FileExistsError):
        write_hpo_configs(base,out,2,"optimizer",17)


def test_run_batch_does_not_execute_hpo_stage_metadata(tmp_path: Path):
    # Exercise orchestration without requiring a valid training run: malformed
    # experiment config must be reported, while HPO_STAGE.json must be ignored.
    cfgdir=tmp_path/"configs";cfgdir.mkdir()
    (cfgdir/"HPO_STAGE.json").write_text(json.dumps({"stage":"optimizer"}))
    (cfgdir/"bad.json").write_text(json.dumps({"not_experiment":True}))
    script=Path(__file__).parents[1]/"scripts"/"run_batch.py"
    proc=subprocess.run([sys.executable,str(script),"--config-dir",str(cfgdir)],capture_output=True,text=True)
    assert proc.returncode==1
    results=json.loads((cfgdir/"batch_results.json").read_text())
    assert len(results)==1
    assert results[0]["config"].endswith("bad.json")
    assert "HPO_STAGE" not in results[0]["config"]

def test_run_batch_preserves_existing_batch_record(tmp_path: Path):
    cfgdir=tmp_path/'configs';cfgdir.mkdir();(cfgdir/'batch_results.json').write_text('[]')
    script=Path(__file__).parents[1]/'scripts'/'run_batch.py'
    proc=subprocess.run([sys.executable,str(script),'--config-dir',str(cfgdir)],capture_output=True,text=True)
    assert proc.returncode!=0
    assert 'batch_results.json already exists' in proc.stderr
    assert (cfgdir/'batch_results.json').read_text()=='[]'
