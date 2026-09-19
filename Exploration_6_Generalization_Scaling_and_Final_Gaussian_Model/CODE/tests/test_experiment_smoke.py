import json
from pathlib import Path
import numpy as np
from puma_exploration6.config import ExperimentConfig
from puma_exploration6.experiment import run_experiment
from puma_exploration6.manifest import write_manifest
from puma_exploration6.utils import sha256_file

CLASSES=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']


def make_artifacts(tmp_path,decoupled=False):
    rows=[]
    # separate ROI/group per split, all classes present; GT coordinates make V17 subset metric defined.
    for split,n_per in [('train',4),('dev',2)]:
        for c in range(10):
            for j in range(n_per):
                rows.append({'uid':f'{split}:{c}:{j}','roi':f'{split}_roi_{c}_{j}','group':f'{split}_g_{c}_{j}','image':f'/missing/{split}_{c}_{j}.tif','x':50.0,'y':50.0,
                             'eval_x':50.0,'eval_y':50.0,'label':c,'class_name':CLASSES[c],'split':split,'coordinate_source':'gt'})
    manifest=tmp_path/'m.csv';write_manifest(manifest,rows)
    rng=np.random.default_rng(1);features=rng.normal(size=(len(rows),1536)).astype('float32');tier=rng.normal(size=(len(rows),16)).astype('float32')
    # add a weak class signal so optimization is nondegenerate
    for i,r in enumerate(rows):features[i,r['label']]+=2
    np.save(tmp_path/'f.npy',features);np.save(tmp_path/'b.npy',tier)
    mh=sha256_file(manifest);(tmp_path/'f.json').write_text(json.dumps({'manifest_sha256':mh}));(tmp_path/'b.json').write_text(json.dumps({'manifest_sha256':mh}))
    cfg={'experiment_id':'smoke_d' if decoupled else 'smoke','manifest':str(manifest),'output_dir':str(tmp_path/'out'),'cached_features':str(tmp_path/'f.npy'),'tier_a':str(tmp_path/'b.npy'),
         'seed':17,'device':'cpu','selection_metric':'roi_macro_f1','sampler':{'mode':'inverse','alpha':1.0},'loss':{'kind':'ce','tau':0},'model':{'representation':'cls','interaction_rank':8},
         'optimizer':{'lr':1e-3,'weight_decay':1e-2,'batch_size':16,'max_epochs':2,'patience':2,'scheduler':'constant','amp':False,'decay_bias_and_vectors':True},
         'decoupled':{'enabled':decoupled,'retrain_epochs':1,'retrain_lr':1e-3,'reset_classifier':True,'sampler':{'mode':'inverse','alpha':1.0},'loss':{'kind':'ce','tau':0}}}
    return ExperimentConfig.from_dict(cfg)


def test_cached_end_to_end_smoke(tmp_path):
    s=run_experiment(make_artifacts(tmp_path));assert s['epochs_run']==2;assert Path(tmp_path/'out/smoke/checkpoints/best.pt').exists();assert np.isfinite(s['best_score'])


def test_decoupled_smoke(tmp_path):
    s=run_experiment(make_artifacts(tmp_path,True));assert s['decoupled'] is not None;assert np.isfinite(s['decoupled']['final_score'])


def test_cached_tokens_global_local_smoke(tmp_path):
    rows=[]
    for split,n_per in [('train',3),('dev',1)]:
        for c in range(10):
            for j in range(n_per):
                rows.append({'uid':f'tok:{split}:{c}:{j}','roi':f'{split}_r_{c}_{j}','group':f'{split}_g_{c}_{j}','image':f'/missing/{split}_{c}_{j}.tif','x':50.25,'y':49.75,
                             'eval_x':50.25,'eval_y':49.75,'label':c,'class_name':CLASSES[c],'split':split,'coordinate_source':'gt'})
    manifest=tmp_path/'tm.csv';write_manifest(manifest,rows);rng=np.random.default_rng(5);tokens=rng.normal(size=(len(rows),265,1536)).astype('float32');tier=rng.normal(size=(len(rows),16)).astype('float32')
    np.save(tmp_path/'tok.npy',tokens);np.save(tmp_path/'tb.npy',tier)
    mh=sha256_file(manifest);(tmp_path/'tok.json').write_text(json.dumps({'manifest_sha256':mh}));(tmp_path/'tb.json').write_text(json.dumps({'manifest_sha256':mh}))
    d={'experiment_id':'tok_gl','manifest':str(manifest),'output_dir':str(tmp_path/'out'),'cached_tokens':str(tmp_path/'tok.npy'),'tier_a':str(tmp_path/'tb.npy'),'device':'cpu','selection_metric':'roi_macro_f1',
       'sampler':{'mode':'inverse','alpha':1.0},'loss':{'kind':'ce','tau':0},'model':{'representation':'global_local','interaction_rank':8},'optimizer':{'batch_size':10,'max_epochs':1,'patience':1,'scheduler':'constant'}}
    s=run_experiment(ExperimentConfig.from_dict(d));assert s['epochs_run']==1 and np.isfinite(s['best_score'])


def test_fixed_final_selection_reaches_last_epoch(tmp_path):
    cfg=make_artifacts(tmp_path);d=cfg.to_dict();d['experiment_id']='fixed';d['optimizer']['selection_mode']='fixed_final';d['optimizer']['max_epochs']=2;d['optimizer']['patience']=1
    s=run_experiment(ExperimentConfig.from_dict(d));assert s['selected_epoch']==2;assert s['selection_mode']=='fixed_final';assert (tmp_path/'out/fixed/checkpoints/final.pt').exists()


def test_completed_run_rejects_same_id_with_different_config(tmp_path):
    cfg=make_artifacts(tmp_path);run_experiment(cfg)
    d=cfg.to_dict();d['optimizer']['lr']=0.002
    import pytest
    with pytest.raises(ValueError,match='different config'):
        run_experiment(ExperimentConfig.from_dict(d))


def test_decoupled_freezes_feature_transform(tmp_path):
    import torch
    cfg=make_artifacts(tmp_path,True);run_experiment(cfg)
    base=torch.load(tmp_path/'out/smoke_d/checkpoints/best.pt',map_location='cpu',weights_only=True)['model']
    crt=torch.load(tmp_path/'out/smoke_d/checkpoints/decoupled_final.pt',map_location='cpu',weights_only=True)['model']
    assert torch.equal(base['head.ph.weight'],crt['head.ph.weight'])
    assert torch.equal(base['head.pb.weight'],crt['head.pb.weight'])
    # Classifier was reset/retrained, so at least the appearance head must differ.
    assert not torch.equal(base['head.head.weight'],crt['head.head.weight'])


def test_epoch_logs_a5_parameter_and_forward_diagnostics(tmp_path):
    s=run_experiment(make_artifacts(tmp_path))
    rec=s['selected']
    assert {'W_weight_norm','W_bias_norm','Ph_weight_norm','Pb_weight_norm','R_weight_norm'} <= set(rec['parameter_norms'])
    assert {'appearance_logit_norm','appearance_projection_norm','biology_projection_norm','interaction_norm','correction_norm'} <= set(rec['dev']['model_diagnostics'])
    assert all(np.isfinite(list(rec['parameter_norms'].values())))


def test_decoupled_uses_deterministic_training_view_contract(tmp_path):
    # Cached mode is already deterministic; this test guards the explicit cRT path
    # and ensures the refactor continues to run with exact same index/exposure logic.
    s=run_experiment(make_artifacts(tmp_path,True))
    d=s['decoupled']['epochs'][0]
    assert d['unique_examples'] > 0 and len(d['exposure']) == 10


def test_full_cached_run_replays_bitwise_with_same_seed(tmp_path):
    import copy, torch
    cfg=make_artifacts(tmp_path)
    d1=cfg.to_dict();d1['experiment_id']='replay_a';d1['output_dir']=str(tmp_path/'out_a')
    d2=copy.deepcopy(d1);d2['experiment_id']='replay_b';d2['output_dir']=str(tmp_path/'out_b')
    s1=run_experiment(ExperimentConfig.from_dict(d1));s2=run_experiment(ExperimentConfig.from_dict(d2))
    assert s1['selected']['selection_score']==s2['selected']['selection_score']
    assert s1['selected']['train']['macro_f1']==s2['selected']['train']['macro_f1']
    assert s1['selected']['dev']['macro_f1']==s2['selected']['dev']['macro_f1']
    c1=torch.load(tmp_path/'out_a/replay_a/checkpoints/final.pt',map_location='cpu',weights_only=True)['model']
    c2=torch.load(tmp_path/'out_b/replay_b/checkpoints/final.pt',map_location='cpu',weights_only=True)['model']
    assert c1.keys()==c2.keys()
    assert all(torch.equal(c1[k],c2[k]) for k in c1)
