import sys,time,json,os,platform
from pathlib import Path
W=Path(__file__).parent;R=W.parent/'outputs/STAGE2_RESEARCH_20260907';sys.path[:0]=[str(W/'deps'),str(R/'01_shared_core')]
import torch,numpy as np,psutil
from torch.utils.data import DataLoader
from dataset import read_manifest,NucleiDataset
from model import load_uni2
def main():
    torch.set_num_interop_threads(1);torch.set_num_threads(4)
    rows=read_manifest(R/'01_sample_definition/sample_manifest.csv');ds=NucleiDataset(rows[:32],96)
    proc=psutil.Process();results={'platform':platform.platform(),'processor':platform.processor(),'logical_cpus':os.cpu_count(),'torch':torch.__version__,'cuda':torch.cuda.is_available(),'loader':[],'forward':[]}
    for workers in [0,2]:
        start=time.perf_counter();n=0
        dl=DataLoader(ds,batch_size=4,num_workers=workers,persistent_workers=workers>0,pin_memory=False)
        for x,_,_ in dl:n+=len(x)
        results['loader'].append({'workers':workers,'seconds':time.perf_counter()-start,'n':n});del dl
    m=load_uni2(r'D:\Research\PUMA\Code\PUMA_pretrained_checkpoints\UNI2-h\uni2_h_model.bin').eval().requires_grad_(False)
    x=torch.stack([ds[i][0] for i in range(4)])
    with torch.inference_mode():
        m(x[:1])
        for threads,batch in [(4,1),(4,2),(4,4),(8,2),(8,4)]:
            torch.set_num_threads(threads);start=time.perf_counter();cpu=time.process_time();z=m(x[:batch]);elapsed=time.perf_counter()-start
            row={'threads':threads,'batch':batch,'seconds':elapsed,'examples_per_second':batch/elapsed,'cpu_seconds':time.process_time()-cpu,'rss_gb':proc.memory_info().rss/1e9,'finite':bool(torch.isfinite(z).all())};results['forward'].append(row);print(row,flush=True)
    results['selected']=max(results['forward'],key=lambda r:r['examples_per_second']);results['loader_selected']=min(results['loader'],key=lambda r:r['seconds'])
    dest=R/'00_reference/performance_profile';dest.mkdir(parents=True,exist_ok=True);(dest/'runtime_profile.json').write_text(json.dumps(results,indent=2));print('DONE',results['selected'],flush=True)
if __name__=='__main__':main()
