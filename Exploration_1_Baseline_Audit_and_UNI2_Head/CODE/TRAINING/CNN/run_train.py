import sys,json,argparse
from pathlib import Path
sys.dont_write_bytecode=True
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--inputs');p.add_argument('--output',required=True);a=p.parse_args();c=json.loads(Path(a.config).read_text())
import original_training_local as t
t.ROOT=Path(a.inputs) if a.inputs else Path(__file__).resolve().parents[4]/'EXPLORATION_1/SHARED_INPUT_SNAPSHOTS/REPRO_INPUTS'
t.OUTPUT_DIR=Path(a.output);t.OUTPUT_DIR.mkdir(parents=True,exist_ok=True);t.RECIPE=c.get('recipe','ce');t.MODES=c.get('modes',['frozen','lora']);sys.argv=[sys.argv[0]]+(['--seed',str(c.get('seed',17))] if c['architecture']=='UNI2_PROBE' else [])
t.main()
