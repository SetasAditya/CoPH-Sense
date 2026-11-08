import copy
from pathlib import Path
import numpy as np
from coph_stagewise.communication_navigation import execute,factory,ROOT
from coph_stagewise.staggered_demo import execute as original_execute
from coph_stagewise.evaluation import load
from coph_stagewise.crossing_demo import configuration
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise.maze_family import NetworkWorld


def test_navigation_manifest_roundtrip(tmp_path):
    from coph_stagewise.communication_navigation import freeze
    a=freeze(tmp_path)
    assert freeze(tmp_path)==a and [s['seed'] for s in a['specs']]==[89500,89501]


def spec(seed,steps=32):
    m=load(ROOT/'results/current_campaign_v1/reciprocity/manifest.json')['protocol']
    return dict(next(s for s in m['specs'] if s['seed']==seed),steps=steps)


def test_disabled_loop_matches_existing_synchronous_and_staggered_execution():
    for seed in (89500,89501):
        s=spec(seed);r=execute(s,'disabled','rotation')
        reference=original_execute(s,'disabled',CommunicationConfig(byte_budget=4000000,ttl=300.),audit=False,
            learner_factory=factory(s),world_factory=NetworkWorld,config=configuration(s['steps']))
        assert r['samples']==reference['samples'] and r['observations']==reference['observations']
        for a,b in zip(r['trajectories'],reference['trajectories']):
            assert len(a)==len(b)
            for p,q in zip(a,b):
                for key in ('x','u','after','version','stage'):assert p[key]==q[key]
        assert [[u['tick'] for u in us] for us in r['updates']]==[[u['tick'] for u in us] for us in reference['updates']]
        assert [[u['diagnostic']['counterfactual_steps'] for u in us] for us in r['updates']]==[[u['diagnostic']['counterfactual_steps'] for u in us] for us in reference['updates']]


def test_selection_off_pairing_and_actual_fixed_rate_transport():
    s=spec(89500)
    a=execute(s,'disabled','rotation');b=execute(s,'disabled','receiver_relevant')
    assert a['samples']==b['samples'] and a['observations']==b['observations']
    for selector in ('rotation','receiver_relevant'):
        r=execute(s,'reciprocal_fixed',selector);sent=[e for e in r['events'] if e['kind']=='sent']
        assert sent and all(e['tick']%8==0 for e in sent)
        assert all(a['max_update_work']<=1024 for a in r['agents'])
        assert all(sum(a['bytes'][k] for k in ('status','payload','acknowledgment'))<=4000000 for a in r['agents'])
        assert all(e['version_after']>=e['version_before'] for e in r['events'] if e['kind']=='delivered')
        if selector=='receiver_relevant':assert sum(a['bytes']['status'] for a in r['agents'])>0
