#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
MPLBACKEND=Agg python - "${1:-results/field_route_v1}" <<'PY'
import json,sys,hashlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
root=Path(sys.argv[1]);rows=json.loads((root/'rows.json').read_text());lookup={(r['case'],r['seed'],r['arm']):r for r in rows}
pairs=[];cases=['clear_fork','direct_risk','bypass_risk'];arms=['anchor_baseline','anchor_reconstruction','direct_local','direct_cooperative']
for case in cases:
 for seed in (78000,78001):
  for left,right in [('anchor_baseline','direct_local'),('anchor_reconstruction','direct_cooperative'),('anchor_baseline','anchor_reconstruction'),('direct_local','direct_cooperative')]:
   a,b=[lookup[case,seed,m] for m in (left,right)]
   pairs.append(dict(case=case,seed=seed,reference=left,candidate=right,agents=[dict(agent=i,risk_cost_delta=b['agents'][i]['components']['risk']-a['agents'][i]['components']['risk'],active_seconds_delta=b['agents'][i]['active_seconds']-a['agents'][i]['active_seconds'],branch_sequence_changed=b['agents'][i]['branch_sequence']!=a['agents'][i]['branch_sequence'],reference_branches=a['agents'][i]['branch_sequence'],candidate_branches=b['agents'][i]['branch_sequence']) for i in (0,1)]))
(root/'paired_comparisons.json').write_text(json.dumps(dict(pairs=pairs,semantics='candidate minus reference, matched case and sensor-noise seed; field assimilation toggles retain remote route anchors and communication',script_sha256=hashlib.sha256(Path('scripts/field_route_pairs.sh').read_bytes()).hexdigest()),indent=2)+'\n')
fig,axes=plt.subplots(2,3,figsize=(14,7),layout='constrained')
for j,case in enumerate(cases):
 for i,color in enumerate(('tab:blue','tab:green')):
  rr=[[lookup[case,s,arm]['agents'][i] for s in (78000,78001)] for arm in arms]
  for ax,metric in [(axes[0,j],lambda a:a['components']['risk']),(axes[1,j],lambda a:a['active_seconds'])]:
   means=[np.mean([metric(a) for a in group]) for group in rr];ax.plot(range(4),means,'o-',color=color,label=f'A{i}')
   for k,group in enumerate(rr):ax.scatter([k,k],[metric(a) for a in group],s=14,color=color,alpha=.5)
 axes[0,j].set_title(case);axes[0,j].set_ylabel('Risk-cost component');axes[1,j].set_ylabel('Active seconds')
 for ax in axes[:,j]:ax.set_xticks(range(4),['anchor local','anchor fused','direct local','direct fused'],rotation=20);ax.legend(fontsize=8)
fig.suptitle('Both agents; two seeds per case. All methods communicate; local/fused toggles received field fitting.')
fig.savefig(root/'figures/paired_agent_outcomes.png',dpi=150);plt.close(fig)
PY
