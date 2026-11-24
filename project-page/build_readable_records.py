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
body = '<p>What was saved for each animation, and what was not recorded.</p>'
for key, record in audit['scenarios'].items():
    body += '<section id="'+key+'"><h2>'+names[key]+'</h2><p>'+cell(record.get('time',''))+'</p>'
    body += table([{'recorded signals': '; '.join(record['available']), 'unavailable signals': '; '.join(record['unavailable']) or 'None listed', 'selection': record['selection_status']}])
    body += '</section>'
body += '<h2>Completed runs</h2>'+table([{'study':k, 'episodes':v['episodes'], 'agent goals':v['agent_goals'], 'incomplete or failed':len(v['incomplete_or_failed'])} for k,v in audit['completion_inventory'].items()])
body += '<h2>Contacts and incomplete runs</h2><p>All '+str(len(audit['failure_inventory']))+' records below include runs with contacts or incomplete coverage. The columns distinguish these outcomes.</p><details><summary>Show all recorded outcomes</summary>'+table(audit['failure_inventory'])+'</details>'
(DIST/'records.html').write_text(page('Recorded evidence',body))


# Keep a single source of truth for the comparative results page.
import runpy
runpy.run_path(str(ROOT / 'build_comparison_figures.py'), run_name='__main__')
