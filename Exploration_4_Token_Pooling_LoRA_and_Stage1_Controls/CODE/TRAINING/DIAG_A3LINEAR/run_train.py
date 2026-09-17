import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import engine
p=argparse.ArgumentParser(description="Run this frozen representation with local code")
p.add_argument('--config',required=True);p.add_argument('--output',required=True)
a=p.parse_args();cfg=json.loads(Path(a.config).read_text());cfg['output_root']=str(Path(a.output).resolve())
from torch import nn
from prompt3_models import Architecture
class Plain(nn.Module):
 def __init__(self,extra=0):super().__init__();self.base=Architecture('A0');self.corrections=[]
 def forward(self,x):return self.base(x[:,:1536])
engine.Model=Plain
engine.run(cfg)
