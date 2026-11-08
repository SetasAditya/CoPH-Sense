#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MPLBACKEND=Agg python - "${1:-results/decision_bottleneck_v1}" <<'PY'
import hashlib,json,sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
root=Path(sys.argv[1]);rows=json.loads((root/'rows.json').read_text());audits=json.loads((root/'audit_rows.json').read_text())
cases=['clear_fork','direct_risk','bypass_risk'];arms=['local_evidence','remote_route','reconstruction','current_replay','historical_replay']
colors=dict(zip(cases,['tab:blue','tab:orange','tab:green']));summary=[];rankings=[]
def branch(nodes):return None if nodes is None else 'bypass' if 4 in nodes or 5 in nodes else 'direct'
for case in cases:
 refs=[b for a in audits if a['case']==case for b in a['references'].values()]
 rankings.append(dict(case=case,audited_branches=len(refs),supported_node_changes=sum(b['supported_reference_changes_route'] for b in refs),full_node_changes=sum(b['full_reference_changes_route'] for b in refs),supported_branch_changes=sum(branch(b['estimated'])!=branch(b['supported_truth']) for b in refs),full_branch_changes=sum(branch(b['estimated'])!=branch(b['full_truth']) for b in refs)))
 for arm in arms:
  rr=[r for r in rows if r['case']==case and r['arm']==arm];aa=[a for r in rr for a in r['agents']]
  summary.append(dict(case=case,arm=arm,episodes=len(rr),goals=sum(a['success'] for a in aa),collisions=sum(a['collision'] for a in aa),timeouts=sum(a['timeout'] for a in aa),risk_cost_mean=[float(np.mean([r['agents'][i]['components']['risk'] for r in rr])) for i in (0,1)],active_seconds_mean=[float(np.mean([r['agents'][i]['active_seconds'] for r in rr])) for i in (0,1)],operational_flow_work=sum(a['replay_work'] for a in aa),policy_changes=sum(r['counts']['committed_policy_changes'] for r in rr),nonzero_gradients=sum(r['counts']['nonzero_gradients'] for r in rr),decision_reaching_paths=sum(r['counts']['decision_reaching_paths'] or 0 for r in rr),returned_paths=sum(r['counts']['returned_paths'] or 0 for r in rr)))
fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
for case in cases:
 aa=[a for a in audits if a['case']==case];margins=[b['margin'] for a in aa for b in a['references'].values() if b['margin'] is not None]
 axes[0].plot(sorted(margins),color=colors[case],label=case.replace('_',' '))
 for arm,marker in [('current_replay','o'),('historical_replay','x')]:
  rr=[r for r in rows if r['case']==case and r['arm']==arm]
  axes[1].scatter([r['counts']['returned_paths'] for r in rr],[r['counts']['decision_reaching_paths'] for r in rr],color=colors[case],marker=marker,s=65,label=f'{case.replace("_"," ")}: {"current" if marker=="o" else "history"}')
 xy=[]
 for a in aa:
  b,d=[a['references'][k]['forces'] for k in ('withhold','deliver')]
  xy.append([np.linalg.norm(np.asarray(d['nominal'])-b['nominal']),np.linalg.norm(np.asarray(d['applied'])-b['applied'])])
 xy=np.asarray(xy);axes[2].scatter(xy[:,0],xy[:,1],s=14,alpha=.5,color=colors[case])
axes[0].set(title='Planner-path score margins',xlabel='Sorted audited branch',ylabel='Second − best score');axes[0].legend(fontsize=8)
axes[1].set(title='Replay decision reach',xlabel='Returned paths per episode',ylabel='Paths entering an outgoing edge');axes[1].legend(fontsize=6)
axes[2].set(title='Packet force transfer',xlabel='Nominal-control change',ylabel='Applied-control change')
axes[2].ticklabel_format(style='sci',axis='both',scilimits=(0,0));axes[2].xaxis.set_major_locator(MaxNLocator(4));axes[2].yaxis.set_major_locator(MaxNLocator(5))
limit=max(axes[2].get_xlim()[1],axes[2].get_ylim()[1]);axes[2].plot([0,limit],[0,limit],'k--',lw=.7,label='No attenuation');axes[2].legend(fontsize=8)
fig.savefig(root/'figures/decision_bottlenecks.png',dpi=150);plt.close(fig)
out=dict(episode_groups=summary,oracle_rankings=rankings,semantics='Two seeds per fixed case. Node-route changes can be connector/waypoint changes on the same branch. Branch classification uses occurrence of bypass nodes 4 or 5 in the remaining route, and does not assert traversal or success.',script_sha256=hashlib.sha256(Path('scripts/decision_summary_figures.sh').read_bytes()).hexdigest())
(root/'summary_aggregates.json').write_text(json.dumps(out,indent=2)+'\n')
PY
