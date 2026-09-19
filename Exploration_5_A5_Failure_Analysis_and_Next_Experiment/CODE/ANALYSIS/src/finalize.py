import pathlib,json,sys,platform,hashlib,time
import pandas as pd,numpy as np
O=pathlib.Path(__file__).resolve().parents[1];A=O.parent/'PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4';M='NOT RECORDED IN ORIGINAL RUN'
def dump(n,v):(O/n).write_text(json.dumps(v,indent=2),encoding='utf-8')
registry=[]
for i,h,output,result in [
 ('P5_D01','H1/H2/H3: can A5 fit training and does validation stall?','A5_LEARNING_DYNAMICS.csv','120 original epochs; gap0.49049 at CV epoch10'),
 ('P5_D02','H5/H4: compare target alignment with frozen feature geometry','REPRESENTATION_GEOMETRY.csv','CLS purity0.2488 vs neighborhood0.4067; class-specific differences'),
 ('P5_D03','H8: quantify class/group concentration','DATA_DIVERSITY_AUDIT.csv','neutrophil/epithelium confirmation effective ROIs1.714/2.118'),
 ('P5_D04','local average gain versus class-specific harm','CONFUSION_CHANGES.csv','epithelium-tumor and neutrophil-apoptosis errors increase'),
 ('P5_D05','H7: Tier-A group association beyond class association','TIER_A_ROI_PERMUTATION.csv','positive group excess in14 non-valid-fraction features; exploratory only'),
 ('P5_D06','do complete proposal sources satisfy upstream outer exclusion?','FULL_BASELINE_ADMISSION.json','no historical fold excluded by every contributing source detector'),
 ('P5_D07','effective exposure is not biological diversity','A5_SAMPLING_EXPOSURE.csv','90 P4 exact-A5 rerun epochs; mean repeat0.3943')]:
 registry.append(dict(experiment_id=i,type='EXPLORATORY_RETROSPECTIVE_DIAGNOSTIC',hypothesis=h,status='completed',new_training_runs=0,output=output,result=result,seed='historical17/29/43; permutation1701 where applicable',protocol='protocol.md',checkpoint=M,new_predictions='none; preserved prediction analysis',promotion_eligible=False))
registry.append(dict(experiment_id='P5_NEXT_A5_FULL',type='PREPARED_NEXT_EXPERIMENT',hypothesis='clean complete-population training coverage improves generalization of retained A5',status='NOT_EXECUTED_ADMISSION_FAILED',new_training_runs=0,output='FULL_DATA_A5_PROTOCOL.md',result='requires verified exclusions and independent partitions',seed='17;29;43',protocol='FULL_DATA_A5_PROTOCOL.md',checkpoint='NOT EXECUTED',new_predictions='NOT EXECUTED',promotion_eligible=False))
pd.DataFrame(registry).to_csv(O/'PROMPT5_EXPERIMENT_REGISTRY.csv',index=False)
decision=dict(baseline='Exploration3_A5',decision='RETAIN_A5',architecture_recommendation='NO NEW ARCHITECTURE IS CURRENTLY JUSTIFIED',dominant_observed_failure='generalization_overfitting',dominant_evidence={'CV_train_macro_F1_epoch10':0.9095012032899681,'CV_mean_held_fold_macro_F1_epoch10':0.4190086273104059,'gap':0.4904925759795622},likely_contributors=['limited_independent_training_and_confirmation_diversity','target_representation_alignment','ROI_association_and_possible_boundary_bias'],causal_decomposition='not established',clean_full_data_baseline_found=False,new_model_training_runs=0,promoted_candidate=None,reused_confirmation_tuning=False,stage1_modified=False,next_experiment={'name':'clean_full_population_retained_A5_with_training_coverage_control','count':1,'parameter_delta':0,'seeds':[17,29,43],'status':'PREPARED_NOT_EXECUTED','protocol':'FULL_DATA_A5_PROTOCOL.md','blockers':['upstream_outer_group_exclusion_not_certified_for_complete_fixed_proposal_mixture','higher_group_identity_unverified','unreused_locked_test_and_four_way_split_not_admitted']},convergence='available diagnostics support next-decision convergence; causal mechanisms remain unresolved',SOTA_claim=False,clinical_claim=False)
dump('PROMPT5_FINAL_DECISION.json',decision)
dump('ENVIRONMENT.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,command_order=['python src/analyze.py','python src/supplement.py','python src/verify_and_localize.py','python src/finalize.py','python src/figures.py'],training_runtime='not executed',diagnostic_runtime_source='ANALYSIS_SUMMARY.json',peak_memory=M))
pd.read_csv(O/'A5_PER_CLASS_STABILITY.csv').query('scope == "CV" and epoch == 10').groupby('class_name')[['recall','F1']].agg(['mean','std','min','max']).to_csv(O/'A5_CLASS_ENDPOINT_SUMMARY.csv')
ledger=pd.read_csv(O/'NEGATIVE_EVIDENCE_LEDGER.csv',na_values=M)
# Fill aggregate P4 semantic scores and paired class harm from source per-class OOF tables.
for table in ['A_complete','D_complete']:
 d=pd.read_csv(A/f'EXPLORATION_4/TABLES/{table}/per_class.csv');base=d[d.family=='A0'].set_index(['seed','class'])
 for fam,g in d.groupby('family'):
  ix=(ledger.prompt==4)&(ledger.experiment_id==fam)&(ledger.experiment_family==table)
  if not ix.any():continue
  delta=(g.set_index(['seed','class']).recall-base.recall).groupby('class').mean().to_dict()
  ledger.loc[ix,'per_class_harm']=json.dumps(delta)
  ledger.loc[ix,'semantic_F1']=float(g.groupby('seed').f1.mean().mean())
ledger.fillna(M).to_csv(O/'NEGATIVE_EVIDENCE_LEDGER.csv',index=False)
lines=['# Negative-evidence lock','', 'These are historical experiments, not Exploration5 reruns. P2 selected epochs differ from P3/P4 fixed10 endpoints. See the report for aggregate interpretation and scope.','', '|Exploration|Experiment|ROI-F1|Delta|Decision|','|---|---|---|---|---|']
for _,r in ledger.iterrows():
 def fmt(v):return f'{v:.6f}' if isinstance(v,(float,int)) and np.isfinite(v) else 'not recorded'
 lines.append(f'|{r.prompt}|{r.experiment_id}|{fmt(r.ROI_F1)}|{fmt(r.delta)}|{r.decision}|')
(O/'NEGATIVE_EVIDENCE_LEDGER.md').write_text('\n'.join(lines),encoding='utf-8')
(O/'findings.md').write_text('# Converged findings\n\nRetain A5. Observed CV training-versus-held semantic F1 gap0.49049. Local-token geometry improves average separability, but confirmation class damage and extreme rare-class ROI concentration prevent promotion. Tier-A has incremental historical signal and measurable group associations; causality is unresolved. No complete-population A5 trial is admitted under the recovered mixed-detector lineage. One next experiment is the clean full-population unchanged-A5 coverage study after admission. No architecture expansion or reused-confirmation rescue.\n',encoding='utf-8')
(O/'research-state.yaml').write_text('phase: converged\nstatus: diagnostic_investigation_complete\nbaseline: Exploration3_A5\nnew_training_runs: 0\nnext_experiment: clean_full_population_A5_coverage\nnext_experiment_status: prepared_admission_failed\nfinal_decision: PROMPT5_FINAL_DECISION.json\n',encoding='utf-8')
required=['EXPLORATION5_ROOT_CAUSE_REPORT.md','NEGATIVE_EVIDENCE_LEDGER.csv','A5_LEARNING_DYNAMICS.csv','A5_PER_CLASS_STABILITY.csv','REPRESENTATION_GEOMETRY.csv','DATA_DIVERSITY_AUDIT.csv','PROMPT5_EXPERIMENT_REGISTRY.csv','PROMPT5_FINAL_DECISION.json']
assert all((O/n).stat().st_size>0 for n in required)
assert len(pd.read_csv(O/'A5_LEARNING_DYNAMICS.csv'))==120
assert len(pd.read_csv(O/'A5_PER_CLASS_STABILITY.csv'))==1200
assert len(ledger)==90
assert set(ledger.prompt)=={2,3,4}
dump('DELIVERABLE_CHECK.json',dict(required_files_present=required,A5_epochs=120,A5_class_epochs=1200,ledger_rows=90,representation_class_rows=len(pd.read_csv(O/'REPRESENTATION_GEOMETRY.csv')),new_training_runs=0))
print('Final deliverables validated')
