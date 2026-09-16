import json,time,shutil
from pathlib import Path
import torch
R=Path(__file__).parent.parent/'outputs/STAGE2_RESEARCH_20260907'
def main():
    torch.set_num_threads(2);torch.manual_seed(17);m=torch.nn.Linear(1536,10).eval();x=torch.randn(64,1536)
    result={'workload':'cached linear classifier, batch64 x1536','cpu_backend':torch.backends.cpu.get_cpu_capability(),'compilers':{n:shutil.which(n) for n in ['cl','g++','clang++']}}
    with torch.inference_mode():
        y=m(x);t=time.perf_counter()
        for _ in range(100):m(x)
        result['eager_100_forward_seconds']=time.perf_counter()-t;t=time.perf_counter()
        try:
            compiled=torch.compile(m,backend='inductor',fullgraph=True);z=compiled(x);result['compile_first_call_seconds']=time.perf_counter()-t
            result['max_abs_difference']=float((z-y).abs().max());t=time.perf_counter()
            for _ in range(100):compiled(x)
            result['compiled_100_forward_seconds']=time.perf_counter()-t;result['status']='measured'
        except Exception as e:result.update(status='unavailable',error_type=type(e).__name__,reason=str(e)[:500],attempt_seconds=time.perf_counter()-t)
    result['decision']='Retain eager; tiny head workloads do not justify compilation setup. This probe does not establish compiled UNI2 throughput.'
    (R/'00_reference/performance_profile/compile_probe.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
