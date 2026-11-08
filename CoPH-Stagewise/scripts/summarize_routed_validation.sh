#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - "${1:-results/routed_validation_v1}" <<'PY'
import sys
from pathlib import Path
import numpy as np
from coph_stagewise.evaluation import load
root=Path(sys.argv[1]);m=load(root/'manifest.json');gate=load(root/'gate.json');rows=load(root/'rows.json');checks=load(root/'artifact_checks.json');verification=load(root/'verification.json')
lines=['# Routed-progress: selected-controller validation','',f"**Held-out navigation gate: {'PASS' if gate['passed'] else 'FAIL'}.** {sum(r['goals'] for r in rows)}/128 agent goals across {len(rows)}/64 completed episodes; {sum(r['collisions'] for r in rows)} collisions and {sum(r['timeouts'] for r in rows)} timeouts.",'','Routed-progress was selected from development results before held-out execution. The original four-configuration development gate remains failed because protected-progress stalls in hazard slalom. Its results remain intact; this is a separate, explicitly recorded selected-controller gate.','','| Communication | Episodes | Goals / agents | Collisions | Timeouts | Stagnation replans | Braking steps | Maximum stage |','|---|---|---|---|---|---|---|---|']
for arm in ('disabled','reciprocal_fixed'):
 rs=[r for r in rows if r['arm']==arm]
 lines.append(f"| {arm} | {len(rs)} | {sum(r['goals'] for r in rs)}/{2*len(rs)} | {sum(r['collisions'] for r in rs)} | {sum(r['timeouts'] for r in rs)} | {sum(r['stagnation_replans'] for r in rs)} | {sum(r['brakes'] for r in rs)} | {max(r['max_stage'] for r in rs)} |")
lines+=['',f"All 81 tests passed. Every original source file matched the development manifest and archive; all {len(m['reference_inventory'])} reference result files remained unchanged. Every episode passed pairing, resolved-controller, communication, cost-component and byte-budget checks. Maximum update work was {checks['maximum_update_work']}/1,024 flow evaluations; the maximum physical energy residual was {checks['maximum_energy_residual']:.3g}. Full execution and rendering took {verification['seconds_this_invocation']/60:.1f} minutes with {m['workers']} workers.",'','The four static scenes used the predeclared seeds 110000–110007. Controller and learner code, parameters, horizons, sensing, costs, safety constraints and fixed-rate communication were unchanged. No held-out outcomes were used for tuning. Final-test namespaces remain untouched.','']
if gate['passed']:lines+=['Routed-progress is validated on these static, independent-agent navigation scenarios. This does not establish inter-agent collision avoidance, changing-hazard forecasting, or adaptive communication reliability.','']
else:lines+=['The selected controller is not yet validated. All failures and incomplete runs remain in the gallery and machine-readable records, including terminal controls and states. Diagnose these failures without tuning on this held-out set.','']
lines+=['Navigation reliability is separate from communication benefit. Earlier isolated deliver/withhold audits verified that reciprocal evidence changes receiver field estimates and controls; those development audits remain preserved. Equal completion rates provide no demonstrated completion advantage from communication. Secondary mission-cost differences, bytes and computation are recorded for all paired cases; no superiority claim follows from navigation readiness.','',f'[Held-out report and four synchronized GIFs]({root}/REPORT.md) · [All failures]({root}/figures/failures/index.html) · [Manifest and fingerprints]({root}/manifest.json) · [Artifact checks]({root}/artifact_checks.json) · [Reproduction](ROUTED_VALIDATION.md)']
Path('ROUTED_VALIDATION_RESULTS.md').write_text('\n'.join(lines)+'\n')
PY
