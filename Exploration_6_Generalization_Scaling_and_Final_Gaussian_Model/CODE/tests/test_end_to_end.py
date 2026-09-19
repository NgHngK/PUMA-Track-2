import csv,json
from puma_exploration6.end_to_end import evaluate_prediction_csv

CLASSES=['tumor','lymphocyte','plasma_cell','histiocyte','melanophage','neutrophil','stroma','epithelium','endothelium','apoptosis']

def test_complete_prediction_csv_perfect(tmp_path):
    gt=tmp_path/'gt';gt.mkdir();features=[];pred=[]
    for c,name in enumerate(CLASSES):
        x=20+25*(c%5);y=20+25*(c//5);ring=[[x-1,y-1],[x+1,y-1],[x+1,y+1],[x-1,y+1],[x-1,y-1]]
        features.append({'type':'Feature','properties':{'classification':{'name':'nuclei_'+name}},'geometry':{'type':'Polygon','coordinates':[ring]}})
        # Evaluator mean includes repeated closing vertex, so use exact serialized mean.
        mx=sum(p[0] for p in ring)/len(ring);my=sum(p[1] for p in ring)/len(ring);pred.append({'roi':'case1','x':mx,'y':my,'score':1.0,'class_name':name,'uid':f'p{c}'})
    (gt/'case1_nuclei.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}))
    pc=tmp_path/'p.csv'
    with pc.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(pred[0]));w.writeheader();w.writerows(pred)
    m=evaluate_prediction_csv(gt,pc);assert abs(m['fixed10']['macro_f1']-1)<1e-12

def test_end_to_end_allows_same_local_uid_in_different_rois(tmp_path):
    import csv, json
    from puma_exploration6.end_to_end import evaluate_prediction_csv
    gt=tmp_path/'gt';gt.mkdir()
    def feature(x,y):
        return {"type":"Feature","properties":{"classification":{"name":"nuclei_tumor"}},"geometry":{"type":"Polygon","coordinates":[[[x-1,y-1],[x+1,y-1],[x+1,y+1],[x-1,y+1],[x-1,y-1]]]}}
    for roi,x in [('a',10),('b',20)]:
        (gt/f'{roi}_nuclei.geojson').write_text(json.dumps({"type":"FeatureCollection","features":[feature(x,10)]}))
    pred=tmp_path/'pred.csv'
    with pred.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['roi','uid','x','y','score','class_name']);w.writeheader()
        w.writerow({'roi':'a','uid':'0','x':10,'y':10,'score':1,'class_name':'tumor'})
        w.writerow({'roi':'b','uid':'0','x':20,'y':10,'score':1,'class_name':'tumor'})
    out=evaluate_prediction_csv(gt,pred)
    assert out['prediction_rows_used']==2
