import sys,csv,json,time,hashlib,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from dataset import read_manifest,CLASSES,centered_crop
from puma_v17_evaluator.data.annotations import load_nuclei_polygons,annotation_centroid
from puma_v17_evaluator.constants import NUCLEUS_CLASSES
def main():
    full=read_manifest(W/'manifest.csv');old=read_manifest(W/'uni2_sample.csv');chosen={r['uid']:r for r in old};rng=np.random.default_rng(17)
    for split,target in [('train',300),('val',150)]:
        pool=[r for r in full if r['split']==split and r['uid'] not in chosen];rng.shuffle(pool)
        while sum(r['split']==split for r in chosen.values())<target:
            current=[r for r in chosen.values() if r['split']==split];counts=collections.Counter(r['label'] for r in current)
            c=min(range(10),key=lambda c:(counts[c],c));positive={r['roi'] for r in current if r['label']==c}
            candidates=[r for r in pool if r['label']==c];assert candidates
            pick=next((r for r in candidates if r['roi'] not in positive),candidates[0]);chosen[pick['uid']]=pick;pool.remove(pick)
    rows=sorted(chosen.values(),key=lambda r:(r['roi'],int(r['uid'].rsplit(':',1)[1])))
    annroot=Path(r'D:\Research\PUMA\Code\TRAINING CODE\Dataset\01_training_dataset_geojson_nuclei');polys={};oracle=[]
    from shapely.geometry import shape
    for r in rows:
        roi=r['roi']
        if roi not in polys:polys[roi]=json.loads((annroot/(roi+'_nuclei.geojson')).read_text())['features']
        feature=polys[roi][int(r['uid'].rsplit(':',1)[1])];geometry=feature['geometry']
        assert feature['properties']['classification']['name']=='nuclei_'+CLASSES[r['label']]
        rings=[geometry['coordinates'][0]] if geometry['type']=='Polygon' else [poly[0] for poly in geometry['coordinates']]
        centers=[annotation_centroid(np.asarray(points,dtype=np.float32)).tolist() for points in rings];center=centers[0]
        r.update(eval_x=float(center[0]),eval_y=float(center[1]),eval_centroids=json.dumps(centers),class_name=CLASSES[r['label']],sampling_seed=17,provenance='prior ROI split; area-centroid crop; V17 exterior-component evaluation')
        p=shape(geometry);v=np.array(p.minimum_rotated_rectangle.exterior.coords);edges=np.linalg.norm(np.diff(v,axis=0),axis=1);major=max(edges);minor=min(edges)
        oracle.append([np.log1p(p.area),p.length,major,minor,minor/max(major,1e-6),4*np.pi*p.area/max(p.length**2,1e-6)])
    dest=R/'01_sample_definition';dest.mkdir(exist_ok=True)
    with (dest/'sample_manifest.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    support={s:{c:{'n':sum(r['split']==s and r['class_name']==c for r in rows),'rois':len({r['roi'] for r in rows if r['split']==s and r['class_name']==c})} for c in CLASSES} for s in ['train','val']}
    meta={'classes':CLASSES,'seed':17,'support':support,'n':len(rows),'rois':len(polys),'manifest_sha256':hashlib.sha256((dest/'sample_manifest.csv').read_bytes()).hexdigest(),'coordinate_source':'GT area-centroid semantic control','eval_centroid':'V17 float32 path-point mean','patient_separation':'unverified','old_cache_overlap':len(old)}
    (dest/'sample_manifest.json').write_text(json.dumps(meta,indent=2))
    cache=R/'02_cache/biology';cache.mkdir(parents=True,exist_ok=True);np.save(cache/'oracle.npy',np.asarray(oracle,dtype=np.float32))
    (cache/'oracle_schema.json').write_text(json.dumps({'tier':'C ORACLE NON-DEPLOYABLE','names':['log1p_GT_area','GT_perimeter','GT_minrect_major','GT_minrect_minor','GT_minrect_axis_ratio','GT_circularity'],'manifest_sha256':meta['manifest_sha256']},indent=2))
    sheet=Image.new('RGB',(5*192,4*216),'white');draw=ImageDraw.Draw(sheet)
    for c in range(10):
        for j,r in enumerate([r for r in rows if r['label']==c][:2]):
            k=2*c+j;x=(k%5)*192;y=(k//5)*216
            with Image.open(r['image']) as im:crop=centered_crop(im.convert('RGB'),r['x'],r['y'],96).resize((192,192))
            sheet.paste(crop,(x,y));draw.text((x+3,y+193),CLASSES[c]+' '+r['split'],fill='black')
    sheet.save(dest/'sample_qc.png');print(json.dumps(meta,indent=2))
if __name__=='__main__':main()
