from pathlib import Path
import csv,json,hashlib,sys,shutil
import numpy as np
from PIL import Image
R=Path(r'C:\Users\Hngk\Documents\Codex'); C=R/'PUMA_STAGE2_PROMPT6/PUMA_STAGE2_PROMPT6_CODE'; P=R/'PUMA_STAGE2_PROMPT6/DATA_PRESELECTION'
O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6'
sys.path.insert(0,str(C/'src'))
from puma_exploration6.manifest import read_manifest
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(p,x): (O/p).write_text(json.dumps(x,indent=2),encoding='utf-8')
def csvout(p,rows):
    with (O/p).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
full=read_manifest(P/'FULL_GT_MANIFEST.csv'); by={r['uid']:r for r in full}
datasets={n:read_manifest(P/f'SELECTION/{n}.csv') for n in ['D300','D600','D900']}
natural=list(csv.DictReader((P/'SELECTION/LOCKED_NATURAL.csv').open(newline='',encoding='utf-8')))
hist={r['roi'] for r in csv.DictReader((R/'PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_3/ARCHITECTURES/A5/INPUT_MANIFESTS/sample_manifest.csv').open(newline='',encoding='utf-8'))}
assert {r['uid'] for r in natural}=={r['uid'] for r in full if r['roi'] not in hist}
prev=set(); support=[]; common={}
for name,rows in datasets.items():
    sets={s:{r['uid'] for r in rows if r['split']==s} for s in ['train','dev','locked']}
    assert prev<=sets['train'];prev=sets['train']
    for s in ['dev','locked']:
        if s in common:assert sets[s]==common[s]
        common[s]=sets[s]
    assert not set(r['roi'] for r in rows)&set(r['roi'] for r in natural)
    for row in rows:
        original=by[row['uid']]
        for k in ['roi','group','image','x','y','label','coordinate_source']:assert row[k]==original[k]
    for split,quota in [('train',[71,37]+[24]*8),('dev',[58,23]+[18]*8),('locked',[59,22]+[18]*8)]:
        if split=='train':quota=[x*int(name[1:])//300 for x in quota]
        for cl in range(10):
            selected=[r for r in rows if r['split']==split and r['label']==cl]
            assert len(selected)==quota[cl]
            groups={g:sum(r['group']==g for r in selected) for g in {r['group'] for r in selected}}
            support.append(dict(dataset=name,split=split,class_id=cl,nuclei=len(selected),groups=len(groups),largest_group_share=max(groups.values())/len(selected),effective_groups=len(selected)**2/sum(x*x for x in groups.values())))
csvout('DATASET_X3/GROUP_SUPPORT_VERIFIED.csv',support)
root=Path(r'D:\Research\PUMA\Code\TRAINING CODE\Dataset'); imgs=root/'01_training_dataset_tif_ROIs'; geo=root/'01_training_dataset_geojson_nuclei'
image_map={}; pairs=[]
for p in imgs.iterdir():
    if p.suffix.lower() in {'.tif','.tiff'}:
        assert p.stem not in image_map;image_map[p.stem]=p
assert set(image_map)=={p.stem.removesuffix('_nuclei') for p in geo.glob('*.geojson')}
for p in sorted(geo.glob('*.geojson')):
    roi=p.stem.removesuffix('_nuclei');im=image_map[roi]
    with Image.open(im) as x:
        x.load();size=x.size
    pairs.append(dict(roi=roi,geojson=str(p),image=str(im),width=size[0],height=size[1],geojson_sha256=sha(p),tiff_sha256=sha(im),status='PASS'))
csvout('DATASET_X3/RAW_ROI_PAIRING.csv',pairs)
hashes={str(p.relative_to(P)):sha(p) for p in P.rglob('*') if p.is_file()};save('DATASET_X3/FROZEN_PRESELECTION_HASHES.json',hashes)
state=json.loads((O/'RESEARCH_STATE/WORKING_STATE.json').read_text());state.update(status='ADMITTED_UNDER_PROSPECTIVE_AMENDMENT',next_required_input=None,new_research_training_runs=0,dataset_admission='PASS',dataset_preselection=str(P),confirmation_semantics='prospectively locked for Exploration 6, but historically exposed to the broader Explorations 1–5 research process',locked_natural_semantics='All ROIs outside the supplied authoritative 181-ROI historical manifest; freshness bounded to that declared exposure source',pending_work=['Complete historical ledger extraction','Verify code QA then deterministic UNI2/Tier-A caching','Controlled A through J experiments','Final ledgers/report/master append'],known_limitations=['PATIENT-LEVEL INDEPENDENCE UNVERIFIED','Natural holdout has zero neutrophil/epithelium support','CPU-only local runtime; compute throughput to be measured'])
save('RESEARCH_STATE/WORKING_STATE.json',state)
save('DATASET_X3/DATA_ADMISSION.json',dict(status='PASS',protocol='Sections 2.7, 15 and 16 of amended exploration6',full_nuclei=len(full),paired_rois=len(pairs),historical_rois=len(hist),locked_natural_nuclei=len(natural),research_train=900,dev=225,confirmation=225,manifest_hashes=hashes,checks=['exact raw identity','fixed holdouts','nested TRAIN','exact class quotas','UID uniqueness','ROI/group disjointness','all unused ROI nuclei reserved','all TIFF pixels readable'],grouping='ROI',patient_independence='UNVERIFIED'))
with (O/'PROVENANCE/CODE_PATCH_LOG.md').open('a',encoding='utf-8') as f:
    f.write('\n## P6-P002: seeded group-ID mapping compatibility\nReproduced 3 CV failures after supplied patch: 74 passed, 3 failed. Installed sklearn 1.7.2 _iter_test_indices shuffles y_counts_per_group in place without remapping groups_inv. On shipped pure-class-group fixture, seed1706 misses classes 8/7 in held folds. Minimal change in src/puma_exploration6/cv.py: seed-permute group IDs, then use StratifiedGroupKFold(shuffle=False). Metadata records exact splitter. This preserves group identity and random tie-breaking; changes internal CV membership, not model/loss/sampler/gradients. Added tests/test_cv_seeded_groups.py to verify class coverage, group exclusion, each sample held once, deterministic replay. Targeted tests: 5 passed. Full suite result recorded separately. No certified reference files changed.\nWindows console issue: preselection --help failed cp1252 encoding of arrow; PYTHONUTF8=1 fixes it without source changes.\n')
shutil.copyfile(Path(__file__),O/'PROVENANCE'/Path(__file__).name)
print(json.dumps(dict(status='PASS',full_nuclei=len(full),rois=len(pairs),locked_natural=len(natural),min_train_class_groups=min(x['groups'] for x in support if x['dataset']=='D900' and x['split']=='train'))))
