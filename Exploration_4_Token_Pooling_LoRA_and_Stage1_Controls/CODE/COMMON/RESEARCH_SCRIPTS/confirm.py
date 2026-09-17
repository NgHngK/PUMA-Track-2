import json
from engine import P,run
assert not json.loads((P/'LORA/decision.json').read_text())['promote']
assert not any(v['promote'] for v in json.loads((P/'BIOMASK_GUIDED/decision.json').read_text()).values())
for seed in [17,29,43]:run({'id':f'FINAL_A3_s{seed}','family':'FINAL_A3','seed':seed,'fold':-1,'epochs':10,'representation':'A3','sampler':'balanced','role':'fixed candidate confirmation on reused original150; no runner-up shopping'})
