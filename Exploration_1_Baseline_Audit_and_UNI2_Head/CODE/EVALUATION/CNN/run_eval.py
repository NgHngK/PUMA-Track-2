import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
from exploration1_train_eval import metrics
p=argparse.ArgumentParser(description="Historical semantic metrics only; V17 was introduced in Exploration2")
p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=np.load(a.predictions);m=metrics(d['labels'],torch.from_numpy(d['logits']));Path(a.output).write_text(json.dumps(m,indent=2));print(m['macro_f1'])
