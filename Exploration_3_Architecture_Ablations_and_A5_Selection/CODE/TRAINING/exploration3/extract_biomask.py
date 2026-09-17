import sys,types,importlib.util,json,csv,hashlib,time,collections
from pathlib import Path
import numpy as np,torch
from PIL import Image
from skimage.draw import polygon
from skimage.measure import regionprops,perimeter
W=Path(__file__).resolve().parent;R=W.parents[1]/'outputs/STAGE2_RESEARCH_20260907';A=R/'00_reference/biomask_audit';S=A/'source';O=R/'02_cache/biomask/prompt3_gt_diagnostic';O.mkdir(parents=True,exist_ok=True)
def load(n,p):
    spec=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(spec);sys.modules[n]=m;spec.loader.exec_module(m);return m
sys.modules['hist']=types.ModuleType('hist');sys.modules['hist.stage2']=types.ModuleType('hist.stage2');load('hist.constants',S/'constants.py')
bio=load('hist.stage2.biomask',S/'biomask.py');cropmod=load('hist.stage2.crops',S/'crops.py');feat=load('hist.stage2.biology_features',S/'biology_features.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
rows=list(csv.DictReader((R/'01_sample_definition/sample_manifest.csv').open()));rm=np.load(A/'roi_manifest.npy');foldmap=dict(zip(rm['roi_id'],map(int,np.load(A/'folds.npy'))))
contract={'manifest_sha256':sha(R/'01_sample_definition/sample_manifest.csv'),'checkpoint_hashes':{str(f):sha(A/f'checkpoints/fold{f}.pt') for f in range(5)},'source_hashes':{n:sha(S/n) for n in ['biomask.py','crops.py','biology_features.py']},'code_hash':sha(Path(__file__)),'coordinate_source':'GT area centroid; historical row-wise fold; upstream-contaminated diagnostic','fov':96,'precision':'FP32','threshold_diagnostic':.5,'uid_order':[r['uid'] for r in rows]}
if (O/'complete.json').exists():
    assert json.loads((O/'contract.json').read_text())==contract;print('Existing BioMask cache complete');sys.exit(0)
(O/'contract.json').write_text(json.dumps(contract,indent=2));torch.set_num_threads(4)
models=[]
for f in range(5):
    m=bio.ProposalBiologyNetwork(32).eval().requires_grad_(False);m.load_state_dict(torch.load(A/f'checkpoints/fold{f}.pt',weights_only=True,map_location='cpu')['model'],strict=True);models.append(m)
n=len(rows);masks=np.zeros((n,96,96),np.float32);features=np.zeros((n,20),np.float32);conf=np.zeros((n,2),np.float32);gtmasks=np.zeros_like(masks,dtype=np.uint8);rgbc=np.zeros((n,96,96,3),np.uint8);quality=[]
groups=collections.defaultdict(list)
for i,r in enumerate(rows):groups[r['roi']].append(i)
def props(mask):
    rr=regionprops(mask.astype('uint8'))
    if not rr:return dict(area=0.,major=0.,minor=0.,eccentricity=0.,perimeter=0.,circularity=0.,cx=float('nan'),cy=float('nan'))
    v=rr[0];p=float(perimeter(mask));return dict(area=float(v.area),major=float(v.axis_major_length),minor=float(v.axis_minor_length),eccentricity=float(v.eccentricity),perimeter=p,circularity=float(4*np.pi*v.area/max(p*p,1e-6)),cx=float(v.centroid[1]),cy=float(v.centroid[0]))
start=time.perf_counter()
for roi,ix in groups.items():
    with Image.open(rows[ix[0]]['image']) as im:rgb=np.array(im.convert('RGB'))
    h=np.clip((-np.log(np.clip(rgb.astype('float32')/255,1/255,1))@np.array([.650,.704,.286],np.float32))/2,0,2);med=float(np.median(h));iqr=max(float(np.percentile(h,75)-np.percentile(h,25)),1e-4)
    annpath=Path(rows[ix[0]]['image']).parent.parent/'01_training_dataset_geojson_nuclei'/(roi+'_nuclei.geojson');annotations=json.loads(annpath.read_text())['features']
    for i in ix:
        r=rows[i];x,y=float(r['x']),float(r['y']);t=cropmod.CropTransform.from_center(x,y,96,rgb.shape);rgbc[i]=t.extract_uint8(rgb);valid=t.valid_region_mask();yy,xx=np.indices((96,96),dtype=np.float32);lx=x-t.crop_left;ly=y-t.crop_top;prompt=np.exp(-((xx-lx)**2+(yy-ly)**2)/(2*2.5**2))
        inp=torch.from_numpy(np.concatenate((rgbc[i].astype('float32').transpose(2,0,1)/255,prompt[None]),axis=0)[None]);vm=torch.from_numpy(valid[None,None])
        with torch.inference_mode():
            z=models[foldmap[roi]](inp,valid_mask=vm);mask=z['mask_logits'].sigmoid()*vm;q=z['quality_logits'].sigmoid();pres=z['presence_logits'].sigmoid();v=feat.extract_biology_features(mask,z['hematoxylin'],z['hematoxylin_gradient'],q,torch.tensor([med]),torch.tensor([iqr]))
        masks[i]=mask[0,0].numpy();features[i]=v[0].numpy();conf[i]=[float(q[0]),float(pres[0])]
        geometry=annotations[int(r['uid'].rsplit(':',1)[1])]['geometry'];polys=[geometry['coordinates']] if geometry['type']=='Polygon' else geometry['coordinates']
        for poly in polys:
            for j,ring in enumerate(poly):
                points=t.roi_to_local(np.array(ring));rr,cc=polygon(points[:,1],points[:,0],shape=(96,96));gtmasks[i,rr,cc]=1 if j==0 else 0
        gtmasks[i]=(gtmasks[i]*valid).astype('uint8');pr=masks[i]>.5;gt=gtmasks[i]>0;inter=(pr&gt).sum();union=(pr|gt).sum();pg=props(gt);pp=props(pr)
        rec={'index':i,'uid':r['uid'],'roi':roi,'class':r['class_name'],'fold':foldmap[roi],'border':bool(valid.mean()<1),'dice':float(2*inter/max(pr.sum()+gt.sum(),1)),'iou':float(inter/max(union,1)),'area_relative_error':float(abs(pp['area']-pg['area'])/max(pg['area'],1)),'centroid_error':float(np.hypot(pp['cx']-pg['cx'],pp['cy']-pg['cy'])),'occupancy':float(pr.mean()),'quality_confidence':float(conf[i,0]),'presence_confidence':float(conf[i,1])}
        rec.update({'gt_'+k:v for k,v in pg.items()});rec.update({'pred_'+k:v for k,v in pp.items()});quality.append(rec)
    if len(quality)%50< len(ix):print('BioMask',len(quality),'/',n,flush=True)
names=['log_area','major','minor','axis_ratio','eccentricity','perimeter','circularity','compactness','H_mean','H_std','gradient_mean','gradient_std','laplacian_mean','entropy_proxy','H_minus_ring','mask_max','mask_mean','quality','robust_z','percentile_proxy']
schema={'names':names,'families':{'morphology':names[:8],'stain_texture':names[8:14],'ring':names[14:15],'reliability':names[15:18],'roi_relative':names[18:]},'status':'predicted features; upstream-contaminated exploratory only','manifest_sha256':contract['manifest_sha256'],'extractor_sha256':sha(S/'biology_features.py'),'GT_quality_not_inputs':True}
np.save(O/'masks.npy',masks);np.save(O/'confidence.npy',conf);np.save(O/'tierB.npy',features);np.save(A/'gt_masks_diagnostic.npy',gtmasks);np.save(A/'rgb_crops_qc.npy',rgbc)
(R/'02_cache/biology/tierB_schema.json').write_text(json.dumps(schema,indent=2));np.save(R/'02_cache/biology/tierB.npy',features)
quality.sort(key=lambda r:r['index'])
with (A/'quality_per_nucleus.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(quality[0]));w.writeheader();w.writerows(quality)
(O/'complete.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'n':n,'mask_sha256':sha(O/'masks.npy'),'feature_sha256':sha(O/'tierB.npy'),'strict_load_all_five':True},indent=2));print('complete seconds',time.perf_counter()-start)
