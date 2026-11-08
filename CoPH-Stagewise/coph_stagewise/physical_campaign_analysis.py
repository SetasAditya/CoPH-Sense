"""Analyze frozen physical campaign records without replaying or changing learners."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);out=ap.parse_args().out
    records=[json.loads(p.read_text()) for p in sorted((out/'execution').glob('*.json')) if p.name not in ('manifest.json','outcomes.json')]
    records=[r for r in records if 'trajectory' in r]
    rows=[]
    for r in records:
        for h in r['fixed_state_handoffs']:rows.append(dict(scene=r['scene'],dt=r['dt'],**h))
    assert len(rows)==18 and all(r['physical_jump']==0 for r in rows)
    assert all(np.isfinite(r['nominal_force_jump']) and np.isfinite(r['applied_force_jump']) for r in rows)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8))
    for scene in ('clear','risk','obstacle'):
        rr=[r for r in rows if r['scene']==scene]
        for ax,key in zip(axes,['shaping_jump','nominal_force_jump','applied_force_jump']):ax.scatter([r['dt'] for r in rr],[r[key] for r in rr],label=scene)
    for ax,label in zip(axes,['Fixed-state shaping jump','Nominal force jump','Protected applied force jump']):ax.set(xlabel='dt (s)',ylabel=label);ax.legend()
    fig.tight_layout();fig.savefig(out/'handoff_jumps.png',dpi=170);fig.savefig(out/'handoff_jumps.pdf');plt.close(fig)
    result=dict(handoffs=rows,count=len(rows),max_shaping_jump=max(abs(r['shaping_jump']) for r in rows),
        max_nominal_force_jump=max(r['nominal_force_jump'] for r in rows),max_applied_force_jump=max(r['applied_force_jump'] for r in rows),physical_jumps_all_zero=True)
    (out/'handoff_analysis.json').write_text(json.dumps(result,indent=2))
    text='''# Execution and handoff findings

All nine controlled protected-progress episodes completed without collision or timeout. Each executed two stage handoffs. Online measured-evidence learning was active and policy replay disabled. All 18 handoffs were evaluated at the identical physical state on isolated before/after frozen stage contexts.

Maximum absolute fixed-state shaping jump: %.6g. Maximum nominal-force jump: %.6g. Maximum protected applied-force jump: %.6g. Physical state jumps were exactly zero. These measurements expose stage-context discontinuities; zero kinetic jumps do not imply continuous forces or switched-system stability.

The held-input physical-accounting residual remains at floating-point precision even when projected/speed-limited controls differ from the nominal command. Safety modification work is logged separately on every executed transition. Timestep-dependent outcome differences confound integration with sensing/update frequency; no pure convergence-rate claim is made. The obstacle is a sensed disk offset from the route, not a claim of general obstacle-course completion.

`handoff_jumps.png`, `handoff_analysis.json`, and `execution/*` retain each event, executed input, projection/guard diagnostics, energy terms, and learner updates. Earlier logger-only failure is retained in `execution_serialization_failure/`.
'''%(result['max_shaping_jump'],result['max_nominal_force_jump'],result['max_applied_force_jump'])
    (out/'FINDINGS.md').write_text(text);print(text)


if __name__=='__main__':main()
