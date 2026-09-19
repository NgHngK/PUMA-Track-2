import pytest
from puma_exploration6.config import ExperimentConfig


def base():
    return {'experiment_id':'x','manifest':'m.csv','output_dir':'out','cached_features':'f.npy','tier_a':'b.npy','sampler':{'mode':'inverse','alpha':1.0},'loss':{'kind':'ce','tau':0},'model':{'representation':'cls'}}


def test_valid_baseline():
    c=ExperimentConfig.from_dict(base());assert c.model.interaction_rank==8


def test_reject_prior_stacking():
    d=base();d['loss']={'kind':'balanced_softmax','tau':1};
    with pytest.raises(ValueError):ExperimentConfig.from_dict(d)


def test_reject_cached_cls_local_representation():
    d=base();d['model']={'representation':'gaussian'}
    with pytest.raises(ValueError):ExperimentConfig.from_dict(d)


def test_encoder_gate_closed_by_default():
    d=base();d.pop('cached_features');d['uni2_weights']='w.pt';d['encoder']={'adaptation':'lora','allow_adaptation':False}
    with pytest.raises(ValueError):ExperimentConfig.from_dict(d)


def test_reject_stain_strength_without_mode():
    d=base();d['augmentation']={'stain_strength':0.05}
    with pytest.raises(ValueError):ExperimentConfig.from_dict(d)


def test_reject_unknown_top_level_config_key():
    d=base();d['weigth_decay']=0.1
    with pytest.raises(ValueError,match='unknown top-level config keys'):
        ExperimentConfig.from_dict(d)


def test_config_roundtrip_is_exact():
    c=ExperimentConfig.from_dict(base())
    assert ExperimentConfig.from_dict(c.to_dict()).to_dict()==c.to_dict()


def test_reject_decoupled_prior_stacking():
    d=base();d['decoupled']={'enabled':True,'sampler':{'mode':'inverse','alpha':1.0},'loss':{'kind':'balanced_softmax','tau':1.0}}
    with pytest.raises(ValueError,match='decoupled'):
        ExperimentConfig.from_dict(d)


def test_reject_jitter_clip_without_sigma():
    d=base();d['augmentation']={'coordinate_jitter_sigma':0.0,'coordinate_jitter_clip':6.0}
    with pytest.raises(ValueError,match='clip must be zero'):
        ExperimentConfig.from_dict(d)


def test_reject_negative_early_stop_min_delta():
    d=base();d['optimizer']={'min_delta':-1e-4}
    with pytest.raises(ValueError,match='min_delta'):
        ExperimentConfig.from_dict(d)


def test_experiment_id_rejects_path_traversal_and_separators():
    import pytest
    from puma_exploration6.config import ExperimentConfig
    base={'experiment_id':'ok','manifest':'m.csv','output_dir':'out','cached_features':'f.npy','tier_a':'b.npy'}
    for bad in ('../escape','a/b','a\\b','..','.',''):
        d=dict(base);d['experiment_id']=bad
        with pytest.raises(ValueError): ExperimentConfig.from_dict(d)
