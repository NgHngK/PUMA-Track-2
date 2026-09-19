import json,datetime
from pathlib import Path
r=Path('2026-09-07/use-the-autoresearch-skill-x20-https'); s=r/'outputs/PUMA_STAGE2_RESEARCH_PROMPTS_1_TO_4/EXPLORATION_6/RESEARCH_STATE'; w=r/'work/exploration6'
note='User disabled automatic autoresearch wakeups to save tokens. Automation puma-stage-2-autoresearch deleted. When authorized code is running, end the chat turn and leave code running; user will reopen chat manually. Do not recreate automation. No training currently running; interrupted AUG0 seed43 remains preserved pending authorized recovery.'
for p in [s/'WORKING_STATE.json',w/'research-state.json']:
 d=json.loads(p.read_text()); d['continuity']=note; d['updated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat(); p.write_text(json.dumps(d,indent=2))
for p in [s/'FINDINGS_CURRENT.md',w/'findings.md']:
 with p.open('a',encoding='utf-8') as f:f.write('\n\nManual continuation preference: '+note+'\n')
(s/'MANUAL_CONTINUATION_PREFERENCE.json').write_text(json.dumps(dict(automatic_wakeups=False,automation_deleted=True,stop_chat_while_code_runs=True,continue_only_on_user_return=True,remaining_B_estimate='Six unstarted runs at8.74–9.49h each:52–57h, plus interrupted seed43 recovery; approximately56–67h if recovery takes4–10h. Full A–J duration remains conditional.'),indent=2))
