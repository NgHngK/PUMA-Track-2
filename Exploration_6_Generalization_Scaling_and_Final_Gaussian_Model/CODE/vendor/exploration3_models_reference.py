"""Finite predeclared architecture stress test. No encoder changes."""
import torch
from torch import nn
class Architecture(nn.Module):
    def __init__(self,kind,dim=16):
        super().__init__();self.kind=kind;self.head=nn.Linear(dim if kind=='B0' else 1536,10);self.branch=None;self.gate=None
        if kind in ['A1','A3','A4','B1','B2','B3','B4']:self.branch=nn.Linear(dim,10,bias=False);nn.init.zeros_(self.branch.weight)
        if kind=='A2':
            self.branch=nn.Sequential(nn.Linear(dim,16),nn.GELU(),nn.Linear(16,10));nn.init.zeros_(self.branch[-1].weight);nn.init.zeros_(self.branch[-1].bias)
        if kind=='A3':self.gate=nn.Parameter(torch.zeros(10))
        if kind=='A4':self.gate=nn.Linear(2,1);nn.init.zeros_(self.gate.weight);nn.init.zeros_(self.gate.bias)
        if kind=='A5':
            self.ph=nn.Linear(1536,8,bias=False);self.pb=nn.Linear(dim,8,bias=False);self.branch=nn.Linear(8,10,bias=False);nn.init.zeros_(self.branch.weight)
        if kind=='A6':self.head=nn.Linear(1536+dim,32);self.output=nn.Sequential(nn.GELU(),nn.Linear(32,10))
    def forward(self,x):
        self.diagnostics={}
        if self.kind=='B0':return self.head(x)
        if self.kind=='A6':
            a=self.head(x);self.diagnostics={'bottleneck_norm':a.detach().norm(dim=1)};return self.output(a)
        h=x[:,:1536];b=x[:,1536:];app=self.head(h)
        if self.kind=='A0':return app
        if self.kind=='A5':
            u=self.ph(h);v=self.pb(b);inter=u*v;delta=self.branch(inter);self.diagnostics.update(interaction_norm=inter.detach().norm(dim=1),appearance_projection_norm=u.detach().norm(dim=1),biology_projection_norm=v.detach().norm(dim=1))
        else:delta=self.branch(b)
        if self.kind=='A3':g=self.gate.sigmoid();delta=g*delta;self.diagnostics['class_gates']=g.detach()
        if self.kind=='A4':
            p=app.detach().softmax(-1);entropy=-(p*p.clamp_min(1e-8).log()).sum(1)/2.302585093;top=p.topk(2).values;g=self.gate(torch.stack((entropy,top[:,0]-top[:,1]),1)).sigmoid();delta=g*delta;self.diagnostics.update(gate=g.detach().squeeze(1),entropy=entropy.detach(),appearance_correct_class=app.detach().argmax(1))
        self.diagnostics['correction_norm']=delta.detach().norm(dim=1)
        if self.kind=='B4':self.diagnostics['tierB_correction_norm']=torch.nn.functional.linear(b[:,16:],self.branch.weight[:,16:]).detach().norm(dim=1)
        return app+delta
