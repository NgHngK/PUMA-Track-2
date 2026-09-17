import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np,torch
from engine import load_data,extra_metrics
p=argparse.ArgumentParser(description="Re-evaluate saved logits under the exact V17 contract")
p.add_argument('--predictions',required=True);p.add_argument('--output',required=True);p.add_argument('--stage1-centers',action='store_true');a=p.parse_args();rows,cv,y=load_data()
from engine import P
ix=json.loads((P/'BIOMASK_GUIDED/REPRESENTATIONS/contract.json').read_text())['original_indices'];rows=[rows[i] for i in ix]
d=np.load(a.predictions);ii=d['indices'];m=extra_metrics([rows[int(i)] for i in ii],d['labels'],torch.from_numpy(d['logits']));Path(a.output).write_text(json.dumps(m,indent=2));print(m['puma']['fixed10']['macro_f1'])
