from bootstrap import *
import datetime
P=R/'EXPLORATION_4';d=R/'EXPLORATION_1/ARCHITECTURES/LORA_EXCLUDED';rows=[]
for p in (d/'RESULTS').rglob('*'):
 if p.is_file():rows.append({'archival_path':str(p.relative_to(R)),'sha256':sha(p),'original_path':'see initial MASTER_ARTIFACT_INDEX.csv; exact executable attribution unresolved','status':'EXCLUDED / NOT EXECUTABLE AS ORIGINALLY RUN'})
if rows:
 csvout(d/'RESULT_INDEX.csv',rows);csvout(d/'ORIGINAL_PATH_MAP.csv',rows)
dump(d/'config.json',{'status':'UNRESOLVED HISTORICAL EXECUTABLE','not_a_recovered_training_configuration':True,'reason':'Exact executed source could not be established. Preserved outputs must not be rerun under an invented implementation.'})
for p in W.glob('*.py'):
 dest=P/'PROVENANCE/RESEARCH_SCRIPTS'/p.name
 if dest.exists() and sha(dest)!=sha(p):dest=dest.with_name(dest.stem+'_closure_'+sha(p)[:8]+dest.suffix)
 cp(p,dest)
cp(W/'report_prompt4.md',P/'PROVENANCE/RESEARCH_SCRIPTS/report_exploration4_source.md')
audit=json.loads((R/'ARCHIVE_VERIFICATION_REPORT.json').read_text());summary=f'''# Archive verification

Original historical files checked: **{audit['original_files_checked']}**. All original and preserved-copy SHA256 values match the initial inventory. No historical file was modified.

Executable architecture packages: **{audit['standalone_tests']}**, all localV17 parity and CLI checks passed. Frozen Exploration4 head packages have strict checkpoint-forward comparisons; the retained A5 package reproduces all three historical confirmation logits with maximum error0. The clean mask checkpoint independently reproduces all156 held-fold probability maps with maximum error0. Test scope is recorded per package; CLI tests are not claimed as retraining.

One excluded historical LoRA architecture has no recovered exact executable source. It remains preserved and explicitly unresolved. Other uncertain historical artifacts remain labeled in the initial inventory; no source association was guessed from a similar filename. The executable-package count excludes this unresolved historical group.

The final inventory is MASTER_ARTIFACT_INDEX_COMPLETE.csv, with all file hashes, result roles and known metadata. Exact duplicate files remain copied for historical preservation and package independence; DUPLICATE_FILE_LINEAGE.json identifies them. Duplicate report blocks and epoch records are separately indexed. The full report includes4,437 unique archived epoch records. Large immutable caches are shared through explicit references, while code is local to each package.

The final selected architecture is retained Exploration3 A5. Exploration4 target-neighborhood pooling failed the declared class-recall confirmation guard despite positive average ROI gains. The study did not execute full-scale classifier training, all-proposal inference, external validation or clinical evaluation. Dataset images and UNI2 foundation weights remain external by documented path and hash.

ZIP verification is recorded separately after compression. Git internals, bytecode and ephemeral download transports are excluded. The downloadable archive contains the complete preserved research tree within this declared scope.
'''
(R/'ARCHIVE_VERIFICATION_REPORT.md').write_text(summary,encoding='utf8')
now=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ');cp(W/'research-state.json',W/f'state_history/{now}.json');s=json.loads((W/'research-state.json').read_text());s.update(status='active',phase='finite experiments complete; A3 confirmation failed; A5 retained; final archive closure',sessions={},completed_runs=sorted(p.parent.name for p in (P/'RUNS').glob('summary.json')) if False else sorted(p.parent.name for p in (P/'RUNS').glob('*/summary.json')),next=['final index and ZIP member verification','commit and completion']);dump(W/'research-state.json',s)
with (W/'findings.md').open('a',encoding='utf8') as f:f.write('\nAll E1 LoRA seeds finished, final argmax predictions identical to frozen controls. Clean mask network completed10epochs; heldDice.783957 and checkpoint inference parity0. Both mask pooling candidates failed promotion. A3 finalconfirmation meanROI gain.01604057 but tumor/neutrophil/epithelium recall drops violate>.10 guard. FINAL: retainExploration3 A5, no runner-up shopping. All3040original hashes unchanged;43executablepackages passparity/CLI; historicalexcludedLoRAunresolved. Masterreportwritten82MB with4437uniqueepochrecords. Archive closure pending.\n')
print('Closure setup complete')
