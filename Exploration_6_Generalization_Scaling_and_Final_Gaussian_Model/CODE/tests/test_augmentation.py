import numpy as np
import torch
from PIL import Image
from puma_exploration6.augmentation import centered_crop_with_token_coordinate,_transform_token_coordinate,image_to_tensor
from puma_exploration6.config import AugmentationConfig


def test_historical_crop_coordinate():
    im=Image.fromarray(np.full((100,100,3),128,dtype=np.uint8));crop,u,v=centered_crop_with_token_coordinate(im,50.25,49.75,96)
    left=int(np.floor(50.25-48+.5));top=int(np.floor(49.75-48+.5));assert crop.size==(96,96);assert u==(50.25-left)/6;assert v==(49.75-top)/6


def test_edge_crop_white_padding_and_center_not_shifted():
    im=Image.fromarray(np.zeros((20,20,3),dtype=np.uint8));crop,u,v=centered_crop_with_token_coordinate(im,.2,.2,96);a=np.asarray(crop)
    assert (a[0,0]==255).all();assert 0<=u<16 and 0<=v<16


def test_four_rotations_return_coordinate():
    u,v=6.3,9.2
    for _ in range(4):u,v=_transform_token_coordinate(u,v,1,False,False)
    assert abs(u-6.3)<1e-5 and abs(v-9.2)<1e-5


def test_image_tensor_deterministic_without_aug():
    im=Image.fromarray(np.arange(96*96*3,dtype=np.uint32).reshape(96,96,3).astype(np.uint8))
    a=image_to_tensor(im,8,8,AugmentationConfig())[0];b=image_to_tensor(im,8,8,AugmentationConfig())[0];torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_fixed_hed_augmentation_is_finite_and_bounded():
    im=Image.fromarray(np.full((96,96,3),[120,70,150],dtype=np.uint8));cfg=AugmentationConfig(stain_mode='fixed_hed',stain_strength=.08);cfg.validate()
    t,u,v,meta=image_to_tensor(im,8,8,cfg,generator=torch.Generator().manual_seed(1));assert torch.isfinite(t).all();assert len(meta['stain_scales'])==2;assert u==8 and v==8
