"""Validate a completed recorded suite without modifying its artifacts."""
import sys
import hashlib
from pathlib import Path
import numpy as np
from coph_stagewise.evaluation import load
from coph_stagewise.visualize import common_probe_error

root=Path(sys.argv[1]);manifest=load(root/'manifest.json')
package=Path(__file__).resolve().parents[1]/'coph_stagewise'
for name,digest in manifest['source_hashes'].items():
    assert hashlib.sha256((package/name).read_bytes()).hexdigest()==digest,name
files=list((root/'episodes').glob('*.json.gz'))
assert len(files)==manifest['expected_episodes']
assert not list((root/'episodes').glob('*.error.json'))
specs={(s['name'],s['seed']):s for s in manifest['scenarios']}
groups={};failures=0;audits=0
for path in files:
    d=load(path);s=d['summary'];key=(s['scene'],s['seed']);groups.setdefault(key,{})[s['arm']]=s
    assert d['spec']==specs[key]
    assert np.isclose(s['mission_cost'],sum(s['components'].values()))
    assert np.isclose(s['mission_cost'],sum(r['cost'] for r in d['trajectory'])+s['components']['failure'])
    assert s['peak_stages']<=d['config']['previous']+1
    assert s['peak_samples']<=s['peak_stages']*d['config']['sample_cap']
    assert len(d['observations'])==s['physical_steps']+1
    assert s['success']+s['collision']+s['timeout']==1
    failures+=not s['success']
    for update in d['updates']:
        assert update['post']['version']>=update['pre']['version']
        assert len(update['probes']['values'])==len(d['probe_points'])
        for ok,value in zip(update['probes']['support'],update['probes']['values']):
            assert ok==(value is not None)
        if not update['diagnostic']['accepted']:
            assert update['pre']==update['post']
    for audit in d['audits']:
        audits+=1
        for branch in audit['results'].values():
            assert branch['states'][0]==audit['start']
            assert len(branch['states'])==audit['horizon']+1
        assert audit['audit_steps']==4*audit['horizon']
    assert len({a['stage'] for a in d['audits']})==len(d['audits'])
    common_probe_error(d,d)
for key,arms in groups.items():
    assert set(arms)==set(manifest['arms'])
    assert arms['full']['counterfactual_steps']==arms['current_replay']['counterfactual_steps']
    for s in arms.values():assert s['useful_replay_steps']+s['padding_steps']==s['counterfactual_steps']
assert all(s['nominal_stage_count']>=8 for s in specs.values() if s['name']=='long_corridor')
print(f'Validated {len(files)} episodes, {audits} audits, {failures} failures; source hashes match.')
if (root/'figures'/'failures'/'index.html').exists():
    from PIL import Image
    assert len(list((root/'figures'/'failures').glob('*.png')))==failures
    gifs=list((root/'figures').glob('*.gif'))
    assert len(gifs)==len({s['name'] for s in manifest['scenarios']})
    for path in gifs:
        with Image.open(path) as im:
            assert im.n_frames>1
            assert im.size==(1200,700)
    print(f'Validated complete failure gallery and {len(gifs)} multiframe animations.')
