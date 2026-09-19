import importlib.util
from pathlib import Path
import torch
from torch.nn import functional as F
from puma_exploration6.config import ModelConfig
from puma_exploration6.models.a5 import A5Head
from puma_exploration6.models.stage2 import Stage2FromCachedCLS
from puma_exploration6.utils import count_trainable_parameters


def _old_architecture():
    p=Path(__file__).parents[1]/'vendor'/'exploration3_models_reference.py'
    spec=importlib.util.spec_from_file_location('old_prompt3_models',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m.Architecture


def test_baseline_parameter_count_exact():
    assert count_trainable_parameters(A5Head())==27866
    assert count_trainable_parameters(A5Head(interaction_rank=4))==21618
    assert count_trainable_parameters(A5Head(interaction_rank=2))==18494


def test_a5_exact_forward_parity_with_prompt3():
    torch.manual_seed(7);Old=_old_architecture();old=Old('A5',16);new=Stage2FromCachedCLS(ModelConfig())
    with torch.no_grad():
        new.head.head.weight.copy_(old.head.weight);new.head.head.bias.copy_(old.head.bias)
        new.head.ph.weight.copy_(old.ph.weight);new.head.pb.weight.copy_(old.pb.weight);new.head.branch.weight.copy_(old.branch.weight)
    raw=torch.randn(11,1536);bio=torch.randn(11,16)
    expected=old(torch.cat((F.layer_norm(raw,(1536,)),bio),1));actual=new(raw,bio)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_zero_initialized_branch_equals_appearance_head():
    torch.manual_seed(9);m=A5Head();h=F.layer_norm(torch.randn(4,1536),(1536,));b=torch.randn(4,16)
    torch.testing.assert_close(m(h,b),m.head(h),rtol=0,atol=0)
