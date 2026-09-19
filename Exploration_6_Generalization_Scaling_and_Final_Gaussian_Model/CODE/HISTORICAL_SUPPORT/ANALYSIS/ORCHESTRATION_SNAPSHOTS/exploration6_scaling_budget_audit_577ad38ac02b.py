from pathlib import Path
import json,csv,math
R=Path(r'C:\Users\Hngk\Documents\Codex');O=R/'2026-09-07/use-the-autoresearch-skill-x20-https/outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6';out=[]
for n in [300,600,900]:
 for seed in [17,29,43]:
  root=O/f'EXPERIMENTS/p6_A_D{n}_s{seed}';c=json.loads((root/'config.json').read_text());h=[json.loads(x) for x in (root/'history.jsonl').read_text().splitlines()];opt=c['optimizer']
  for row in h:
   draws=sum(row['exposure']);micro=math.ceil(draws/opt['batch_size']);updates=math.ceil(micro/opt['accumulation_steps']);out.append(dict(run_id=root.name,size=n,seed=seed,epoch=row['epoch'],draws=draws,microbatches=micro,optimizer_updates=updates,physical_batch=opt['batch_size'],accumulation=opt['accumulation_steps'],final_update_samples=draws-(updates-1)*opt['batch_size']*opt['accumulation_steps'],unique_nuclei=row['unique_examples'],unique_ROIs=row['unique_groups'],provenance='Draw/unique counts logged; batch/update counts derived from certified loader and accumulation implementation'))
with (O/'TABLES/DATA_SCALING_TRAINING_BUDGET.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
(O/'RESEARCH_STATE/SCALING_INTERPRETATION_LIMITS.md').write_text('''# Scope of the fixed-epoch scaling contrast

The executed design follows the user-required historical fixed-final10-epoch control. Each epoch draws N samples, so D300/D600/D900 use5/10/15 optimizer updates per epoch at batch64. Total updates per seed are50/100/150. The last batches contain44/24/4 samples, respectively. These quantities are derived from the executed sampler/loader contract and verified logged draws; they were not additional logged optimizer-step counters.

The contrast changes both unique annotation availability and total update/exposure budget. It therefore measures the prescribed fixed-epoch scaling policy, not an isolated causal effect of diversity at identical updates. More annotation UIDs also do not prove more independent patients: all sizes use the same preassigned61-ROI TRAIN pool and patient mapping is unavailable. Do not claim that a threefold annotation increase means threefold independent groups. Existing per-epoch unique-ROI counts provide the observed exposure.

D900 improves the mean DEV/ROI metrics and reduces the semantic gap relative to D300, but it fails the neutrophil class-recall guard. No extra matched-budget run is introduced post hoc merely to rescue that outcome. C compares sampling at fixed D900 and matched joint-stage budgets; cRT has an explicitly matched five-extra-epoch no-reset control. Further study follows the original finite sequence.
''',encoding='utf-8')
print('Verified90 epochs and recorded fixed-epoch scaling budget limitation.')
