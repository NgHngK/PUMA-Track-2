import json, zipfile, collections, hashlib, platform
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(r'D:\Research\PUMA\Code\TRAINING CODE')
OUT=Path(__file__).parent/'ingestion'; OUT.mkdir(parents=True,exist_ok=True)
for p in Path(r'C:\Users\Hngk\Downloads').glob('*V17*.docx'):
    with zipfile.ZipFile(p) as z:
        xml=ET.fromstring(z.read('word/document.xml'))
        ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        paragraphs=[''.join(n.itertext()) for n in []]
        paragraphs=[''.join(t.text or '' for t in el.findall('.//w:t',ns)) for el in xml.findall('.//w:p',ns)]
        (OUT/(p.stem+'.txt')).write_text('\n'.join(paragraphs),encoding='utf-8')
with zipfile.ZipFile(ROOT/'PUMA_nuclei_biology_audit.zip') as z:
    (OUT/'zip_inventory.json').write_text(json.dumps([{'name':i.filename,'bytes':i.file_size} for i in z.infolist()],indent=2))
counts=collections.Counter(); coverage=collections.defaultdict(set); rois=[]; examples=[]
for p in sorted((ROOT/'Dataset/01_training_dataset_geojson_nuclei').glob('*.geojson')):
    obj=json.loads(p.read_text()); fs=obj.get('features',[]) if isinstance(obj,dict) else obj
    c=collections.Counter()
    for f in fs:
        label=f.get('properties',{}).get('classification',{}).get('name','UNKNOWN')
        counts[label]+=1;c[label]+=1;coverage[label].add(p.stem)
    rois.append({'roi':p.stem.removesuffix('_nuclei'),'count':len(fs),'classes':dict(c)})
    if len(examples)<2: examples.append(fs[0] if fs else {})
audit={'counts':dict(counts),'coverage':{k:len(v) for k,v in coverage.items()},'rois':rois,'examples':examples}
(OUT/'annotation_audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps({k:v for k,v in audit.items() if k!='rois'},indent=2)[:7000])
print('ROIs',len(rois),'TIFs',len(list((ROOT/'Dataset/01_training_dataset_tif_ROIs').glob('*.tif'))))
print('checkpoint bytes',(ROOT/'pytorch_model.bin').stat().st_size)
