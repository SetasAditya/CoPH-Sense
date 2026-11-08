#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - "${1:-results/completion_first_v1}" <<'PY'
import hashlib,sys
from pathlib import Path
from dataclasses import replace
import torch
from coph_stagewise.evaluation import load,dump
from coph_stagewise.records import Config
from coph_stagewise.model import tensor,potential,shaping_parameters
root=Path(sys.argv[1]);rows=[]
for scene,seed,mode in [('asymmetric_coverage',s,'protected') for s in range(61000,61008)]+[('blocked_risk_passage',61000,'routed')]:
    path=Path('results/navigation_fix_v3/development/episodes')/f'{scene}_{seed}_{mode}_reciprocal_fixed.json.gz'
    r=load(path);snap=r['updates'][0][-1]['post'];st=snap['stages'][-1];t=r['trajectories'][0][-1];x=tensor(t['x']);target=tensor(t.get('navigation',{}).get('waypoint',r['spec']['goals'][0]));cfg=replace(Config(**r['config']),shaping_mode='progress');w=tensor(st['weights']);centers=tensor(st['centers']);policy=tensor(snap['policy']);obs=snap['geometry'];values={}
    for label,c in [('legacy',replace(cfg,shaping_mode='legacy')),('progress',cfg)]:
        q=x[:2].clone().requires_grad_(True);g=torch.autograd.grad(potential(q,target,w,centers,c.rbf_width,policy,obs,c),q)[0];values[label]=dict(radial_gradient=float((q-target)@g),force=(-g).tolist())
    lower=.5*float((x[:2]-target).square().sum());assert values['progress']['radial_gradient']>=lower-1e-10
    rows.append(dict(scene=scene,seed=seed,mode=mode,state=x.tolist(),target=target.tolist(),recorded_nominal=t['navigation']['nominal'],model_version=snap['version'],trajectory_version=t['version'],context='last recorded post-update model; fixed state and target, not a rerun',bound=shaping_parameters(w,cfg.rbf_width,obs,cfg),required_radial_lower_bound=lower,**values))
dump(root/'recorded_cancellation_checks.json',dict(rows=rows,script_sha256=hashlib.sha256(Path('scripts/check_completion_regressions.sh').read_bytes()).hexdigest()))
print([(r['mode'],r['seed'],r['legacy']['radial_gradient'],r['progress']['radial_gradient']) for r in rows])
PY
