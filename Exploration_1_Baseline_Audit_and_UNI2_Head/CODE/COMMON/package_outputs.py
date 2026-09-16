import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).parent.parent;OUT=ROOT/'outputs';WORK=ROOT/'work'
paths=[]
for p in OUT.rglob('*'):
    if p.is_file() and p.suffix in {'.py','.md','.txt','.png','.pdf'} and '__pycache__' not in p.parts:paths.append(p)
names=['analyze_results.py','build_report.py','cpu_controls.py','discrimination.py','extract_uni2.py',
       'lora_dynamics.py','probe_uni2.py','plot_results.py','test_integrity.py','report_body.md',
       'PROTOCOL.md','UNI2_PROTOCOL.md','LORA_PROTOCOL.md','REPLICATION_PROTOCOL.md','DISCRIMINATION_PROTOCOL.md',
       'research-state.yaml','research-log.md','findings.md','protocol_git_log.txt',
       'manifest.csv','manifest.json','cnn_sample.csv','cnn_sample.json','uni2_sample.csv','uni2_sample.json',
       'uni2_features.npy','uni2_complete.npy','uni2_features.json','integrity_checks.json',
       'results_summary.json','bootstrap.json','discrimination.json','crop_contact_sheet.png']
paths.extend(WORK/n for n in names)
for directory in ['cnn_results','uni2_results','lora_results','lora_results_v2']:
    paths.extend(p for p in (WORK/directory).glob('*') if p.suffix in {'.json','.jsonl','.npz','.pt'} and p.name!='prefix.pt')
paths.extend(WORK/'ingestion'/n for n in ['annotation_audit.json','history_compact.csv','drive_history.csv','drive_preprocessing_guide.md','zip_inventory.json'])
paths.extend((WORK/'literature').glob('*.md'))
assert all(p.is_file() for p in paths)
manifest={str(p.relative_to(ROOT)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(set(paths))}
dest=OUT/'PUMA_STAGE2_RESEARCH_BUNDLE.zip'
with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p in sorted(set(paths)):z.write(p,p.relative_to(ROOT))
    z.writestr('BUNDLE_MANIFEST.json',json.dumps(manifest,indent=2))
with zipfile.ZipFile(dest) as z:
    assert z.testzip() is None
    for name,entry in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==entry['sha256']
print(json.dumps({'archive':str(dest),'files':len(paths)+1,'bytes':dest.stat().st_size,'crc_and_hashes':'PASS'},indent=2))
