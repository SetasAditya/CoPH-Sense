"""Verify saved arrays, returns, provenance, and paired initial conditions."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

out = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'results/native_v1'
manifest = json.loads((out/'manifest.json').read_text())
metrics = json.loads((out/'metrics.json').read_text())
assert hashlib.sha256((out/'runner.py.txt').read_bytes()).hexdigest()==manifest['runner_sha256']
assert len(metrics['rows'])==40
for row in metrics['rows']:
    arrays = np.load(out/('%s_%02d.npz'%(row['protocol'],row['episode'])))
    assert len(arrays['positions']) == row['steps']+1
    assert arrays['actions'].shape==(row['steps'],4,1,2)
    assert np.isfinite(arrays['actions']).all()
    assert np.isfinite(arrays['positions']).all()
    rewards=arrays['rewards'].reshape(-1,4)
    assert row['summed_agent_return']==sum(float(r.sum()) for r in rewards)
    assert row['mean_agent_return']==sum(float(r.mean()) for r in rewards)
    assert row['collision_pair_steps']==int(arrays['collisions'].sum())
    assert row['steps']==manifest['protocols'][row['protocol']]
for i in range(10):
    original=np.load(out/('upstream_original_%02d.npz'%i))
    legacy=np.load(out/('wrapper_legacy_check_%02d.npz'%i))
    corrected=np.load(out/('corrected_400_%02d.npz'%i))
    random=np.load(out/('random_400_%02d.npz'%i))
    for key in original.files:
        assert np.array_equal(original[key],legacy[key])
    for other in [corrected,random]:
        assert np.array_equal(original['positions'][0],other['positions'][0])
        assert np.array_equal(original['landmarks'],other['landmarks'])
    assert np.array_equal(original['positions'],corrected['positions'][:201])
    assert np.array_equal(original['actions'],corrected['actions'][:200])
with Image.open(out/'navigation_episode0.gif') as gif:
    assert gif.n_frames==101
checks=dict(episodes=40,bitwise_original_wrapper_episodes=10,
    bitwise_original_corrected_200_step_prefixes=10,paired_initial_states=True,
    reward_recomputation=True,finite_arrays=True,manifest_runner_hash=True,
    gif_frames=101)
(out/'artifact_checks.json').write_text(json.dumps(checks,indent=2))
print(json.dumps(checks,indent=2))
