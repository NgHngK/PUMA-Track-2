import torch
from torch.nn import functional as F
from puma_exploration6.models.pooling import TokenPooler,gaussian_pool,target_patch_pool,neighborhood_pool


def tokens():
    torch.manual_seed(1);return torch.randn(3,265,1536)


def test_shapes_and_finite():
    t=tokens();u=torch.tensor([8.0,7.25,9.9]);v=torch.tensor([8.0,8.8,6.1])
    for kind in ['cls','target_patch','gaussian','neighborhood','global_local','scalar_global_local']:
        p=TokenPooler(kind);x=p(t,u,v);assert x.shape==(3,1536);assert torch.isfinite(x).all()


def test_target_patch_exact_index():
    t=torch.zeros(1,265,1536);t[0,9+5*16+7]=3
    out=target_patch_pool(t,torch.tensor([7.9]),torch.tensor([5.2]));assert torch.all(out==3)


def test_neighborhood_radius1_is_3x3_at_interior():
    t=torch.zeros(1,265,1536);sp=t[:,9:].reshape(1,16,16,1536);sp[:,7:10,7:10]=9
    out=neighborhood_pool(t,torch.tensor([8.1]),torch.tensor([8.2]),1);assert torch.all(out==9)


def test_global_local_exact_prompt4_a4_equation():
    t=tokens();u=torch.tensor([8.,7.5,9.]);v=torch.tensor([8.,8.5,7.]);g=gaussian_pool(t,u,v,1.5)
    expected=F.layer_norm((F.layer_norm(t[:,0],(1536,))+F.layer_norm(g,(1536,)))/2,(1536,))
    actual=TokenPooler('global_local',1.5)(t,u,v)
    torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-6)


def test_gaussian_weights_preserve_constant_field():
    t=torch.zeros(2,265,1536);t[:,9:]=4.25
    out=gaussian_pool(t,torch.tensor([1.,14.]),torch.tensor([1.,14.]),1.5);torch.testing.assert_close(out,torch.full_like(out,4.25))
