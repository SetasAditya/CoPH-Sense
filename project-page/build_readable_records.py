"""Render archived gallery data as static, accessible HTML tables."""
from pathlib import Path
from html import escape
import json
import shutil

ROOT = Path(__file__).resolve().parent
DIST = ROOT / 'dist'

def load(name):
    return json.loads((DIST / 'assets' / name).read_text())

def cell(value):
    if value is None: return 'Unavailable'
    if isinstance(value, bool): return 'Yes' if value else 'No'
    if isinstance(value, float): return f'{value:.4g}'
    if isinstance(value, (dict, list)): return '; '.join(f'{k}: {cell(v)}' for k, v in value.items()) if isinstance(value, dict) else '; '.join(map(cell, value))
    return escape(str(value))

def table(rows, columns=None):
    if not rows: return '<p>No records.</p>'
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    return '<div class="table-wrap"><table><thead><tr>' + ''.join('<th scope="col">'+escape(k.replace('_',' ').capitalize())+'</th>' for k in columns) + '</tr></thead><tbody>' + ''.join('<tr>'+''.join('<td>'+cell(row.get(k))+'</td>' for k in columns)+'</tr>' for row in rows)+'</tbody></table></div>'

def page(title, body):
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+title+' · CoPH</title><link rel="stylesheet" href="style.css"><header class="topbar"><a class="brand" href="index.html">CoPH</a><a href="index.html#scenarios">Back to scenarios</a></header><main><section><h1>'+title+'</h1>'+body+'</section></main></html>'

audit = load('log_audit.json')
provenance = load('provenance.json')
names = {'dynamics':'Dynamics change','hazards':'Different hazard preferences','reciprocity':'Reciprocal discovery','encounters':'Symmetric encounters','uncertainty':'Sparse evidence'}
body = '<p>Recorded signals and presentation timing for the five animations. Missing signals are listed explicitly.</p>'
for key, record in audit['scenarios'].items():
    body += '<section id="'+key+'"><h2>'+names[key]+'</h2><p>'+cell(record.get('time',''))+'</p>'
    body += table([{'recorded signals': '; '.join(record['available']), 'unavailable signals': '; '.join(record['unavailable']) or 'None listed', 'selection': record['selection_status']}])
    frames = provenance['frames'][key]
    cols = [k for k in frames[0] if 'sha256' not in k]
    body += '<details><summary>Frame timing table ('+str(len(frames))+' frames)</summary>'+table(frames, cols)+'</details></section>'
body += '<h2>Completed cohorts</h2>'+table([{'study':k, 'episodes':v['episodes'], 'agent goals':v['agent_goals'], 'incomplete or failed':len(v['incomplete_or_failed'])} for k,v in audit['completion_inventory'].items()])
body += '<h2>Adverse records</h2><p>All '+str(len(audit['failure_inventory']))+' archived adverse records are retained below. A contact or incomplete coverage can both place an episode in this inventory; these are not all the same failure type.</p><details><summary>Open the complete adverse-record table</summary>'+table(audit['failure_inventory'])+'</details>'
(DIST/'records.html').write_text(page('Recorded evidence',body))

body = '<p>Results from the archived studies. The scenario tabs link directly to the corresponding figure or table.</p>'
for key, filename, caption in [
 ('dynamics','fig11_drag_shift.png','After a mid-mission drag change, online identification adapts while a fixed model becomes stale. Curves, paired returns and identification traces use the original study results.'),
 ('native','fig12_native_heldout.png','The original 50-start held-out comparison, with matched allocation, targets and protection. This cohort is distinct from the later 100-start evaluation.'),
 ('uncertainty','fig13_route_certificate.png','Saved-state path choices from identical evidence: route regret, uncertainty coverage and wrong-choice rates. These are not navigation episodes.')]:
    src = ROOT.parent.parent/'CoPH-Stagewise'/'paper_figures'/'output'/filename
    shutil.copyfile(src,DIST/'assets'/filename)
    body += '<section id="'+key+'"><h2>'+{'dynamics':'Surprise and recovery','native':'Matched native comparison','uncertainty':'Route uncertainty'}[key]+'</h2><figure><a href="assets/'+filename+'"><img class="result-figure" src="assets/'+filename+'" alt="'+escape(caption)+'" loading="lazy"></a><figcaption>'+caption+'</figcaption></figure></section>'
cert = json.loads((ROOT.parent/'docs/results/route_certificate_v1/summary.json').read_text())
body += table([{'representation':k,'choices certified':f'{100*v["certified_share"]:.1f}%', 'wrong among certified':f'{100*v["wrong_when_certified"]:.1f}%', 'true gap inside nominal 90% interval':f'{100*v["calibration90"]:.1f}%'} for k,v in cert.items()])
body += '<section id="hazards"><h2>Hazard decisions</h2><p>Original four-scene pilot. Carrier worst-case excess cost is measured against the best arm in the same scene; smaller is better. Raw readings match the learned field, and receiver fitting adds no demonstrated navigation benefit here.</p>'+table([{'shared content':name,'worst-case carrier excess cost':value} for name,value in [('Learned field',4.6),('Raw readings',4.6),('Field without receiver fitting',4.6),("Scout’s avoid/free decision",25.3),('No communication',68.1)]])+'</section>'
body += '<section id="reciprocity"><h2>Reciprocal discovery</h2><p>In the displayed controlled crossing, reciprocal warnings save 27.1 m of team detour relative to no communication. This is a layout-specific result.</p><img class="result-figure" src="assets/reciprocity.png" alt="Logged crossing trajectories and discovery diagnostics"></section>'
body += '<section id="encounters"><h2>Protected encounters</h2>'+table([{'evaluation':'Final layer, each encounter family','completion':'30/30','contacts':0},{'evaluation':'Old layer, four-agent swap','completion':'0/30','contacts':'See full study records'}])+'<p>Yielding is programmed. Priority can increase crossing time; it is not evidence of learned strategy discovery.</p><img class="result-figure" src="assets/encounters.png" alt="Logged base and final protection encounter comparison"></section>'
(DIST/'results.html').write_text(page('Figures and results',body))
print('Built results.html and records.html from archived records.')
