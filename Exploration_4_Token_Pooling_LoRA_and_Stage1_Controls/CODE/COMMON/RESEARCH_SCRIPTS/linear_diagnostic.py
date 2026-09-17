import engine as e
from torch import nn
from prompt3_models import Architecture
class Plain(nn.Module):
 def __init__(self,extra=0):super().__init__();self.base=Architecture('A0');self.corrections=[]
 def forward(self,x):return self.base(x[:,:1536])
if __name__=='__main__':
 e.Model=Plain
 for s in [17,29,43]:
  for f in range(3):e.run({'id':f'DIAG_A3LINEAR_s{s}_cv{f}','family':'DIAG_A3LINEAR','seed':s,'fold':f,'epochs':10,'representation':'A3','sampler':'balanced','selection_eligible':False})
