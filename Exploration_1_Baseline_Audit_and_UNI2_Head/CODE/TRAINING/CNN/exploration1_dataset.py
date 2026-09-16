"""Stage-2 crops only. All coordinates are level-0 ROI pixel x,y; Stage 1 is external."""
import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, WeightedRandomSampler

CLASSES = ['tumor','lymphocyte','plasma_cell','histiocyte','melanophage',
           'neutrophil','stroma','epithelium','endothelium','apoptosis']
MEAN = torch.tensor([.485,.456,.406])[:,None,None]
STD = torch.tensor([.229,.224,.225])[:,None,None]

def read_manifest(path):
    path = Path(path)
    with path.open(newline='',encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    required = {'uid','roi','group','image','x','y','label','split','coordinate_source'}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f'Manifest must contain {sorted(required)}')
    # Prefer human-readable class names; existing manifests require an explicit sidecar.
    if 'class_name' not in rows[0]:
        sidecar=path.with_suffix('.json')
        if not sidecar.exists() or json.loads(sidecar.read_text()).get('classes')!=CLASSES:
            raise ValueError('Manifest requires class_name column or canonical classes in JSON sidecar')
    seen = set(); group_split = {}; roi_split = {}; image_split = {}
    for r in rows:
        if r['uid'] in seen: raise ValueError('Duplicate uid: '+r['uid'])
        seen.add(r['uid'])
        if not r['image']: raise ValueError('Empty image')
        p = Path(r['image'])
        r['image'] = str((p if p.is_absolute() else path.parent/p).resolve())
        for key, ledger in [('group',group_split),('roi',roi_split),('image',image_split)]:
            if not r[key]: raise ValueError(f'Empty {key}')
            value = str(Path(r[key])).casefold() if key=='image' else r[key]
            if value in ledger and ledger[value] != r['split']:
                raise ValueError(f'{key} leakage: {r[key]}')
            ledger[value] = r['split']
        if r['split'] not in {'train','val','calibration','test','predict'}:
            raise ValueError('Unknown split')
        if r['coordinate_source'] not in {'gt','stage1_oof','stage1_frozen'}:
            raise ValueError('Unknown coordinate provenance')
        r['label'] = int(r['label']); r['x'] = float(r['x']); r['y'] = float(r['y'])
        if r['label'] not in range(10) and not (r['split']=='predict' and r['label']==-1):
            raise ValueError('Label outside canonical ten-class ontology')
        if 'class_name' in r and r['label']>=0 and r['class_name']!=CLASSES[r['label']]:
            raise ValueError('Integer label and class name disagree')
        if not np.isfinite([r['x'],r['y']]).all(): raise ValueError('Nonfinite center')
        if not Path(r['image']).is_file(): raise FileNotFoundError(r['image'])
    return rows

def centered_crop(image, x, y, fov):
    """Preserve requested center near edges; pad missing context white, never shift inward."""
    if fov < 2: raise ValueError('fov must be >=2 pixels')
    w,h = image.size
    if not (0 <= x < w and 0 <= y < h): raise ValueError('Center outside ROI')
    left,top = int(np.floor(x-fov/2+.5)),int(np.floor(y-fov/2+.5))
    box=(max(left,0),max(top,0),min(left+fov,w),min(top+fov,h))
    out=Image.new('RGB',(fov,fov),(255,255,255))
    out.paste(image.crop(box),(box[0]-left,box[1]-top))
    return out

def image_tensor(crop, output_size=224, augment=False, stain_strength=0.):
    crop=crop.resize((output_size,output_size),Image.Resampling.BICUBIC)
    a=np.array(crop,dtype=np.float32)/255.
    if augment:
        a=np.rot90(a,int(torch.randint(4,()).item()))
        if torch.rand(())<.5: a=a[:,::-1]
        if stain_strength:
            # Optional RGB optical-density perturbation; not stain deconvolution.
            scales=1+stain_strength*(2*torch.rand(3).numpy()-1)
            a=np.exp(np.log(np.clip(a,1/255,1))*scales)
    t=torch.from_numpy(a.copy()).permute(2,0,1)
    return (t-MEAN)/STD

class NucleiDataset(Dataset):
    def __init__(self, rows, fov=96, output_size=224, augment=False, stain_strength=0., cache_rois=2):
        self.rows=rows; self.fov=fov; self.output_size=output_size
        self.augment=augment; self.stain_strength=stain_strength
        self.cache_rois=cache_rois; self.cache=OrderedDict()
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; path=r['image']
        if path not in self.cache:
            with Image.open(path) as im: self.cache[path]=im.convert('RGB')
            while len(self.cache)>self.cache_rois: self.cache.popitem(last=False)
        self.cache.move_to_end(path)
        crop=centered_crop(self.cache[path],r['x'],r['y'],self.fov)
        return image_tensor(crop,self.output_size,self.augment,self.stain_strength),r['label'],i

def sampling(rows, mode='natural', seed=17):
    labels=torch.tensor([r['label'] for r in rows])
    counts=torch.bincount(labels,minlength=10).double()
    if (counts==0).any(): raise ValueError('Training lacks classes: '+str([CLASSES[i] for i in range(10) if counts[i]==0]))
    if mode=='natural': return None,counts/counts.sum()
    if mode!='balanced': raise ValueError(mode)
    weights=1/counts[labels]
    q=torch.zeros(10,dtype=torch.float64).scatter_add_(0,labels,weights); q/=q.sum()
    return WeightedRandomSampler(weights,len(rows),replacement=True,generator=torch.Generator().manual_seed(seed)),q

def make_manifest(root, out, group_csv=None):
    """GT centers are a clean semantic control; never claim them as Stage-1 detections."""
    from shapely.geometry import shape
    from sklearn.model_selection import StratifiedGroupKFold
    root=Path(root); rows=[]
    groups={}
    if group_csv:
        with open(group_csv,newline='',encoding='utf-8') as f:
            groups={r['roi']:r['patient_id'] for r in csv.DictReader(f)}
    for p in sorted((root/'01_training_dataset_geojson_nuclei').glob('*.geojson')):
        roi=p.stem.removesuffix('_nuclei'); image=root/'01_training_dataset_tif_ROIs'/(roi+'.tif')
        if not image.is_file(): raise FileNotFoundError(image)
        if group_csv and roi not in groups: raise ValueError('Missing patient mapping: '+roi)
        obj=json.loads(p.read_text()); features=obj['features'] if isinstance(obj,dict) else obj
        for i,f in enumerate(features):
            name=f['properties']['classification']['name'].removeprefix('nuclei_')
            if name not in CLASSES: raise ValueError(name)
            g=shape(f['geometry'])
            if g.is_empty or not g.is_valid or g.area<=0: raise ValueError(f'Invalid nucleus polygon {roi}/{i}')
            center=g.centroid
            rows.append(dict(uid=f'{roi}:{i}',roi=roi,group=groups.get(roi,roi),image=str(image.resolve()),
                             x=center.x,y=center.y,label=CLASSES.index(name),class_name=name,split='',coordinate_source='gt'))
    if not rows: raise ValueError('No GeoJSON nuclei found')
    # A fixed, reproducible development split. This does not preserve legacy fold membership.
    split=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=17)
    fold=np.zeros(len(rows),dtype=int)
    for k,(_,ix) in enumerate(split.split(np.zeros(len(rows)),[r['label'] for r in rows],[r['group'] for r in rows])):
        fold[ix]=k
    for r,k in zip(rows,fold): r['split']='val' if k==0 else 'train'
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    meta={'classes':CLASSES,'grouping':'patient' if group_csv else 'ROI ONLY; patient separation unverified',
          'source':'GT-centered semantic development control','seed':17,
          'counts':{s:np.bincount([r['label'] for r in rows if r['split']==s],minlength=10).tolist() for s in ['train','val']}}
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);p.add_argument('--group-csv')
    a=p.parse_args();make_manifest(a.root,a.out,a.group_csv)
