#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/coph-mpl
python - "${1:-results/routed_validation_v1}" <<'PY'
import sys,hashlib
from pathlib import Path
from PIL import Image
from coph_stagewise.evaluation import load,dump
from coph_stagewise.routed_validation import readiness,verify_sources,inventory,verify_episode,ARMS
root=Path(sys.argv[1]);m=load(root/'manifest.json');reference=Path(m['reference'])
assert inventory(reference)==m['reference_inventory']
assert verify_sources(reference)==m['source_verification']
completed=load(root/'completion.json');assert len(completed)==64
specs={(s['name'],s['seed']):s for s in m['specs']};rows=[]
for c in completed:
 if not c['complete']:continue
 r=load(root/'episodes'/c['path']);verify_episode(r,specs[c['scene'],c['seed']],c['arm'],m['controller'],m['communication'])
 assert c['metrics']['goals']==sum(a['success'] for a in r['agents'])
 assert c['metrics']['collisions']==sum(a['collision'] for a in r['agents'])
 assert c['metrics']['timeouts']==sum(a['timeout'] for a in r['agents'])
 assert c['metrics']['max_update_work']<=1024
 assert c['metrics']['max_energy_residual']<1e-10
 rows.append(c['metrics'])
failures=load(root/'failures.json');assert len(failures)==sum(not c['complete'] or c.get('metrics',{}).get('goals',0)!=2 for c in completed)
assert len(list((root/'figures/failures').glob('*.png')))==sum(c['complete'] and c['metrics']['goals']!=2 for c in completed)
media=[]
for name in sorted({s['name'] for s in m['specs']}):
 p=root/'figures'/f'{name}.gif';f=load(root/'figures'/f'{name}_frames.json')
 with Image.open(p) as image:assert image.n_frames==len(f['ticks'])==32;media.append(dict(scene=name,frames=image.n_frames,size=image.size))
 assert f['seed']==110000
 rs=[load(root/'episodes'/f'{name}_110000_routed_{a}.json.gz') for a in ARMS]
 end=max(t['tick'] for r in rs for seq in r['trajectories'] for t in seq)
 assert f['ticks'][0]==0 and f['ticks'][-1]==end
assert readiness(completed,m['specs'])==load(root/'gate.json')
assert len(load(root/'paired.json'))==32
result=dict(reference_unchanged=True,episodes=len(completed),completed=len(rows),media=media,failures=len(failures),gate=load(root/'gate.json'),maximum_update_work=max(r['max_update_work'] for r in rows),maximum_energy_residual=max(r['max_energy_residual'] for r in rows),final_test_executed=False,script_sha256=hashlib.sha256(Path('scripts/check_routed_validation.sh').read_bytes()).hexdigest())
dump(root/'artifact_checks.json',result);print(result)
PY
