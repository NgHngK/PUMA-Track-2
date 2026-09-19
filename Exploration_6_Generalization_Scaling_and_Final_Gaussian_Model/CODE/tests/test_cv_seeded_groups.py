from puma_exploration6.cv import create_internal_group_cv_manifests
from puma_exploration6.manifest import write_manifest, read_manifest
from puma_exploration6.constants import CLASSES


def test_seeded_pure_class_groups_preserve_coverage_and_replay(tmp_path):
    rows = [dict(uid=f'{c}_{g}_{i}', roi=f'{c}_{g}', group=f'{c}_{g}',
                 image=f'{c}_{g}.tif', x=1, y=1, label=c,
                 class_name=name, split='train', coordinate_source='gt')
            for c,name in enumerate(CLASSES) for g in range(6) for i in range(2)]
    source=tmp_path/'source.csv'
    write_manifest(source,rows)
    memberships=[]
    for attempt in range(2):
        folds=create_internal_group_cv_manifests(source,tmp_path/f'cv{attempt}',seed=1706)
        held=[]
        for fold in folds:
            loaded=read_manifest(fold['manifest'],require_files=False)
            train=[r for r in loaded if r['split']=='train']
            dev=[r for r in loaded if r['split']=='dev']
            assert {r['label'] for r in dev}==set(range(10))
            assert not {r['group'] for r in train}&{r['group'] for r in dev}
            held.append(sorted(r['uid'] for r in dev))
        assert sorted(x for fold in held for x in fold)==sorted(r['uid'] for r in rows)
        memberships.append(held)
    assert memberships[0]==memberships[1]
