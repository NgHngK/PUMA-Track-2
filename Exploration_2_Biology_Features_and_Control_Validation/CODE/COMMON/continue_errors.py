import sys,json,collections
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path.insert(0,str(R/'01_shared_core'))
from dataset import read_manifest,CLASSES,centered_crop
def main():
    rows=[r for r in read_manifest(R/'01_sample_definition/sample_manifest.csv') if r['split']=='val'];out=R/'91_error_analysis';out.mkdir(exist_ok=True)
    a=np.load(R/'experiments/exp_loss_balanced_ce/predictions/best.npz');b=np.load(R/'experiments/exp_fusion_tierA/predictions/best.npz');assert np.array_equal(a['uids'],b['uids'])
    y=a['labels'];pa=a['logits'].argmax(1);pb=b['logits'].argmax(1);assert [r['uid'] for r in rows]==a['uids'].tolist()
    groups={'UNI wrong / biology correct':np.flatnonzero((pa!=y)&(pb==y)),'UNI correct / biology wrong':np.flatnonzero((pa==y)&(pb!=y)),
       'both wrong':np.flatnonzero((pa!=y)&(pb!=y)),'both correct':np.flatnonzero((pa==y)&(pb==y))}
    report={k:{'n':len(ix),'uids':[rows[i]['uid'] for i in ix],'true_class_counts':dict(collections.Counter(CLASSES[y[i]] for i in ix))} for k,ix in groups.items()}
    (out/'disagreement.json').write_text(json.dumps(report,indent=2))
    sheet=Image.new('RGB',(5*192,4*236),'white');draw=ImageDraw.Draw(sheet)
    for row,(name,ix) in enumerate(groups.items()):
        for col,i in enumerate(ix[:5]):
            x=col*192;top=row*236;r=rows[i]
            with Image.open(r['image']) as im:crop=centered_crop(im.convert('RGB'),r['x'],r['y'],96).resize((192,192))
            sheet.paste(crop,(x,top));draw.ellipse((x+93,top+93,x+99,top+99),outline='cyan',width=1)
            draw.text((x+2,top+193),name,fill='black');draw.text((x+2,top+206),'GT: '+CLASSES[y[i]],fill='black');draw.text((x+2,top+219),f'U: {CLASSES[pa[i]]} B: {CLASSES[pb[i]]}',fill='black')
    sheet.save(out/'disagreement_contact_sheet.png')
    displacement=np.array([np.hypot(float(r['eval_x'])-r['x'],float(r['eval_y'])-r['y']) for r in rows]);report2={'label':'annotation area center versus first V17 exterior vertex mean; NOT Stage1 displacement','median':float(np.median(displacement)),'p90':float(np.percentile(displacement,90)),'p95':float(np.percentile(displacement,95)),'maximum':float(displacement.max()),'n':len(rows)}
    (out/'annotation_center_displacement.json').write_text(json.dumps(report2,indent=2))
    print({k:v['n'] for k,v in report.items()})
if __name__=='__main__':main()
