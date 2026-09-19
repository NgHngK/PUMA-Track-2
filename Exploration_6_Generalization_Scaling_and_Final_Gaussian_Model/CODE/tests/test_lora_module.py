import torch
from puma_exploration6.models.uni2 import QVLoRA


def test_qv_lora_zero_update_exact_base_parity():
    torch.manual_seed(1);base=torch.nn.Linear(12,36);x=torch.randn(2,5,12);expected=base(x).detach().clone();m=QVLoRA(base,rank=4,alpha=8);actual=m(x)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_qv_lora_gradient_connectivity_and_k_unchanged():
    torch.manual_seed(2);base=torch.nn.Linear(8,24);m=QVLoRA(base,rank=2,alpha=4);x=torch.randn(3,4,8,requires_grad=True);z=m(x);q,k,v=z.chunk(3,-1);loss=(q.square().mean()+v.square().mean());loss.backward()
    assert m.bq.grad is not None and m.bv.grad is not None
    assert torch.count_nonzero(m.bq.grad)>0 and torch.count_nonzero(m.bv.grad)>0
    with torch.no_grad():base_k=base(x.detach()).chunk(3,-1)[1];torch.testing.assert_close(k.detach(),base_k,rtol=0,atol=0)
