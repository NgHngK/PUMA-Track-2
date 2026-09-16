import sys,json,time,hashlib,collections
from pathlib import Path
import numpy as np,psutil
from PIL import Image
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from dataset import read_manifest
from biology import SCHEMA,NAMES,roi_channels,point_features
def main():
    manifest=R/'01_sample_definition/sample_manifest.csv';rows=read_manifest(manifest);groups=collections.defaultdict(list)
    for i,r in enumerate(rows):groups[r['roi']].append(i)
    result=np.zeros((len(rows),len(NAMES)),dtype=np.float32);times={'load':0.,'channels':0.,'point_features':0.};roi_stats={};start=time.perf_counter();cpu=time.process_time();proc=psutil.Process();peak=0
    for roi,ix in groups.items():
        t=time.perf_counter()
        with Image.open(rows[ix[0]]['image']) as im:rgb=np.asarray(im.convert('RGB'))
        times['load']+=time.perf_counter()-t;t=time.perf_counter();ch=roi_channels(rgb);times['channels']+=time.perf_counter()-t;t=time.perf_counter()
        for i in ix:result[i]=point_features(ch,rows[i]['x'],rows[i]['y'])
        times['point_features']+=time.perf_counter()-t;roi_stats[roi]={'median':ch['median'],'iqr':ch['iqr']};peak=max(peak,proc.memory_info().rss)
    cache=R/'02_cache/biology';cache.mkdir(parents=True,exist_ok=True);np.save(cache/'tierA.npy',result)
    schema={'tier':'A','families':SCHEMA,'names':NAMES,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256((R/'01_shared_core/biology.py').read_bytes()).hexdigest(),'coordinate_source':'GT-centered control; extractor deployable from RGB and any valid point','spatial':'excluded: no verified frozen Stage1 coordinate set','radius_center':12,'ring_radii':[16,24]}
    (cache/'tierA_schema.json').write_text(json.dumps(schema,indent=2))
    stats=R/'02_cache/roi_stats';stats.mkdir(parents=True,exist_ok=True);(stats/'stain_stats.json').write_text(json.dumps(roi_stats))
    elapsed=time.perf_counter()-start;report={'timings':times,'total_seconds':elapsed,'cpu_seconds':time.process_time()-cpu,'peak_sampled_rss_gb':peak/1e9,'rois':len(groups),'n':len(rows),'examples_per_second':len(rows)/elapsed,'roi_per_second':len(groups)/elapsed,'optimization':'ROI-open once; channels once; vectorized point measurements; bounded one-ROI memory'}
    dest=R/'00_reference/performance_profile';dest.mkdir(parents=True,exist_ok=True);(dest/'biology_profile.json').write_text(json.dumps(report,indent=2));print(report)
if __name__=='__main__':main()
