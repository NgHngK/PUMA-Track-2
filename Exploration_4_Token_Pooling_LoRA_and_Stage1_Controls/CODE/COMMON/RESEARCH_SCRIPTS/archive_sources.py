from bootstrap import *
sources=[]
for n,src in [(2,OLD/'00_reference/MASTER_PROMPT.md'),(3,OLD/'00_reference/EXPLORATION3.md')]:cp(src,R/f'PROMPT_{n}/EXPLORATION_TEXT/original_exploration.md')
p=R/'EXPLORATION_1/EXPLORATION_TEXT/REQUEST_RECONSTRUCTION.md'
if not p.exists():p.write_text('''# Exploration1 request reconstruction

The original request was supplied in the conversation, not as a recoverable original Exploration1 file. This is a labeled reconstruction, not a verbatim artifact.

The user requested the autoresearch loop for fixed-Stage1, ten-class PUMA nuclei classification; ingestion of the supplied reports, ratio file, GeoJSON/TIFF examples and archive; primary-source literature grounding; rigorous internal reviewer/architect reasoning; finite CPU/GPU probes with loss, gradients, tail recall and validation dynamics; and a detailed Stage2 Markdown report with dataset/model/loss/train-evaluation code. Stage1 detection and tissue segmentation were outside the modification scope. Subsequent user messages supplied the verified UNI2-h checkpoint path and authorized the continuation files. Exact surviving reports and source artifacts are preserved with hashes; missing original prompt text is not invented.
''',encoding='utf8')
paths=[(1,Path('D:/Research/PUMA/Code/TRAINING CODE/PUMA_nuclei_biology_audit.zip')),(1,Path('D:/Research/PUMA/Code/TRAINING CODE/Biology Nuclei Feature Report.pdf')),(1,Path('D:/Research/PUMA/Code/TRAINING CODE/data ratio.txt')),(1,Path('C:/Users/Hngk/Downloads/PUMA_Stage2_Research_Report_Debate_Experiments_V17.docx')),(1,Path('C:/Users/Hngk/Downloads/PUMA_V17_Research_Debate_Experiment_Report.docx')),(2,Path('C:/Users/Hngk/Downloads/prompt.md')),(3,Path('C:/Users/Hngk/Downloads/exploration3.md')),(3,Path('D:/Research/PUMA/Code/Biomask training.txt')),(4,Path('C:/Users/Hngk/Downloads/exploration4.md'))]
for n,src in paths:
 if not src.exists():sources.append({'source':str(src),'destination':'','sha256':'','status':'MISSING ORIGINAL'});continue
 dest=R/f'PROMPT_{n}/PROVENANCE/USER_SUPPLIED_ORIGINALS'/src.name;cp(src,dest);sources.append({'source':str(src),'destination':str(dest.relative_to(R)),'sha256':sha(src),'status':'VERIFIED COPY'})
csvout(R/'USER_ASSET_LINEAGE.csv',sources)
# Retain executed research code, without ephemeral downloads or signed transport URLs.
for p in W.glob('*.py'):cp(p,R/'EXPLORATION_4/PROVENANCE/RESEARCH_SCRIPTS'/p.name)
for p in [W/'protocol.md',W/'alignment_protocol.md',W/'diagnostic_protocol.md']:cp(p,R/'EXPLORATION_4/PROVENANCE/RESEARCH_SCRIPTS'/p.name)
print('Original documents and research scripts archived')
