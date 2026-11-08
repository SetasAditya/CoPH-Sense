"""Read-only access to the frozen native phMARL comparison (benchmarks/phmarl/results/native_safety_v2)."""
import gzip
import json
from pathlib import Path
import numpy as np

DEFAULT=Path(__file__).resolve().parents[2]/'benchmarks'/'phmarl'/'results'/'native_safety_v2'
ARMS=('phmarl','attraction','attraction_protected','coph_frozen_protected','coph_online_protected','phmarl_protected')
LABEL=dict(phmarl='Published phMARL',attraction='Attraction',attraction_protected='Attraction + protection',
           coph_frozen_protected='Frozen CoPH + protection',coph_online_protected='Online CoPH + protection',phmarl_protected='phMARL + protection')
COLOR=dict(phmarl='#4D4D4D',attraction='#B0B5BB',attraction_protected='#E69F00',coph_frozen_protected='#56B4E9',coph_online_protected='#8E24AA',phmarl_protected='#009E73')
SEEDS=tuple(range(81000,81020))
PRIOR=1-np.exp(-.6*.1)  # controller's initial discrete drag
NATIVE_DRAG=.25          # VMAS simple_spread velocity drag


def load(root=DEFAULT):
    """Per-arm, per-seed metrics and arrays; asserts agreement with the frozen summary."""
    root=Path(root);summary=json.loads((root/'metrics.json').read_text())['summary'];out={};sources=[]
    for arm in ARMS:
        rows=[]
        for s in SEEDS:
            mp=root/f'{arm}_seed{s}_metrics.json';zp=root/f'{arm}_seed{s}.npz';sources+=[mp,zp]
            m=json.loads(mp.read_text());z=np.load(zp)
            rewards=z['rewards'][:,:,0].astype(float)
            # Arrays are float32; they must reproduce the logged float64 metrics to rounding.
            assert abs(rewards.sum(0).mean()-m['mean_agent_return'])<1e-3,(arm,s)
            assert int(z['contact_pairs'].sum())==round(m['collision_pair_steps']),(arm,s)
            rows.append(dict(seed=s,metrics=m,positions=z['positions'],landmarks=z['landmarks'],rewards=rewards,contacts=z['contact_pairs']))
        mean=np.mean([r['metrics']['mean_agent_return'] for r in rows])
        assert abs(mean-summary[arm]['mean_agent_return'])<1e-9,(arm,mean)
        out[arm]=rows
    return out,sources


def drag_traces(root=DEFAULT):
    """Committed drag estimate per (seed, agent) as a step function over the 400 steps."""
    root=Path(root);traces=[];sources=[]
    for s in SEEDS:
        p=root/f'coph_online_protected_seed{s}_updates.json.gz';sources.append(p)
        with gzip.open(p,'rt') as h:updates=json.load(h)
        for agent in sorted({u['agent'] for u in updates}):
            value=np.full(401,PRIOR);ups=sorted((u for u in updates if u['agent']==agent),key=lambda u:u['step'])
            for u in ups:value[u['step']:]=u['committed']
            traces.append(value)
    return np.array(traces),sources
