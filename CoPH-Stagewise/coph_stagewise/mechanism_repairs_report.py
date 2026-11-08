"""Aggregate completed isolated repairs without revising old experiment claims."""
import argparse,hashlib,json,shutil
from pathlib import Path
from PIL import Image


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args();out=args.out.resolve()
    package=Path(__file__).resolve().parents[1]
    route=read(out/'route_decision/results.json');comm=read(out/'communication/final_summary.json')
    review=read(out/'route_decision/independent_verification.json')
    local=read(out/'reconstruction/verification.json')
    communications=read(out/'communication/ARTIFACT_VERIFICATION.json')
    snapshots=read(out/'route_decision/manifest.json')
    for name,digest in snapshots['sources'].items():assert sha(package/'coph_stagewise'/name)==digest,name
    for rel,digest in snapshots['inputs'].items():assert sha(package/rel)==digest,rel
    frames=[]
    for p in sorted((out/'route_decision/figures').glob('*.gif')):
        with Image.open(p) as im:
            assert im.n_frames==60;frames.append(dict(path=str(p.relative_to(out)),frames=im.n_frames,size=list(im.size),sha256=sha(p)))
    assert len(frames)==3 and len(route['rows'])==24
    assert all(a['success'] and not a['collision'] for r in route['rows'] for a in r['agents'])
    assert all(r['maximum_update_flow']<=1024 for r in route['rows'])
    reports=['route_decision/REPORT.md','route_decision/INDEPENDENT_REVIEW.md','reconstruction/REPORT.md',
        'reconstruction/ORIGINAL_QUERY_REPORT.md','communication/FINDINGS.md','overlap/final_v2/REPORT.md']
    checks=dict(route_episodes=24,route_agent_goals=48,route_collisions=0,route_timeouts=0,route_source_hashes_current=True,
        preserved_snapshot_inputs_unchanged=True,animations=frames,independent_route_checks=review,
        localized_checks=local,communication_checks=communications,
        reports={r:sha(out/r) for r in reports},report_script_sha256=sha(Path(__file__)))
    full_log=next(p for p in ['/tmp/coph-mechanism-full-tests-release.log','/tmp/coph-mechanism-full-tests-final.log','/tmp/coph-mechanism-full-tests-correct-cwd.log'] if Path(p).exists())
    for source,name in [(full_log,'tests.log'),('/tmp/coph-mechanism-native-tests.log','native_tests.log'),
        ('/tmp/coph-mechanism-full-tests.log','incorrect_cwd_attempt.log')]:
        if Path(source).exists():shutil.copyfile(source,out/name)
    tests=(out/'tests.log').read_text() if (out/'tests.log').exists() else 'unavailable'
    checks['test_result_tail']=tests.strip().splitlines()[-1:]
    nav=out/'communication/navigation_v2/REPORT.md'
    if nav.exists():
        checks['communication_navigation_report_sha256']=sha(nav)
        checks['communication_navigation_verification']=read(out/'communication/navigation_v2/ARTIFACT_VERIFICATION.json')
        checks['selector_source_amendment']=read(out/'communication/SELECTOR_AMENDMENT.json')
        movie=out/'communication/navigation_v2/figures/selected_comparison.gif'
        with Image.open(movie) as im:
            assert im.n_frames==40
            checks['communication_navigation_animation']=dict(frames=im.n_frames,sha256=sha(movie))
    (out/'verification.json').write_text(json.dumps(checks,indent=2))
    text='''# Mechanism repairs: implementation and evidence

Six proposed changes are implemented as opt-in mechanisms. Original defaults, old results, failed eligibility gates and final-test namespaces are preserved. These results establish tested mechanism changes, not that every original superiority claim now holds.

| Mechanism | Implemented change | Evidence | Remaining limitation |
|---|---|---|---|
| Route-choice prediction | Explicit bounded route candidates, supported physical prefixes, approximate graph continuation; current/historical starts | 24 saved snapshots and 24 new unchanged-fork episodes; 48/48 goals, zero collisions/timeouts | Terminal variants match graph outcomes; no demonstrated historical-replay benefit |
| Localized reconstruction | Local-only RBF proposal plus bounded compact residual; actual measured-data validation and transactional rollback | 184 snapshots, 552 matched updates; 146,970 exact outside-neighborhood value/gradient equalities | Helps the current fixed-query set but is worse than global fitting on the older set |
| Compact message transport | Exact full-message JSON+zlib binary frame; decode and normal validation | 36 matched-evidence payloads, exact roundtrips | Controlled estimation/byte comparison, not general navigation/compression superiority |
| Receiver-relevant payloads | Advertised route/support stage selection at fixed transmission opportunities | 108 corrected nonempty selection pairs; 24 real transport/update checks; 16 navigation episodes | No mission benefit in the navigation pilot; advertisements are expensive |
| Local consistency gate | Corrected-receiver fault reaudits, then separately frozen local residual quarantine | 2,560 initial +1,280 valid gated updates on 160 preserved snapshots | Rejects useful evidence; remains off by default |
| Smooth overlap | Differentiate complete potential blend over supported retained models | Fixed-support handoff potential/force jumps 0; replay FD error 2.04e-11 | Hard support and context changes remain; no general stability/passivity theorem |

## Route decisions and historical replay

All four arms use the same scene specifications, noise seeds, controller, fitting, validation, communication schedule, support and 1,024-flow update ceiling. Policy-coefficient replay is disabled to isolate route-choice prediction. There are three unchanged symmetric-fork cases, two reused development seeds and four arms. They are engineering comparisons; prior decision-sensitivity gates remain failed and are not reinterpreted as passed.

'''
    text+='| Arm | Goals /12 | Mean full cost | Valid candidate batches | Flow work |\n|---|---:|---:|---:|---:|\n'
    for name,s in route['summary'].items():text+=f"| {name} | {s['goals']} | {s['mean_cost']:.6f} | {s['valid_decisions']}/{s['all_decisions']} | {s['flow']} |\n"
    text+='''
Prefix-only scoring changes branches in four paired runs and increases cost. Adding approximate remaining-route cost restores the graph baseline's outcomes; the current and historical terminal variants have exactly the baseline mission costs and actual branch choices. This demonstrates the need to account for truncated consequences in this comparison, not a gain over the existing graph planner.

On the 24 preserved decision snapshots, current-state candidate batches are valid in 8/24 and historical batches in 0/24. Unsupported alternatives invalidate the batch and retain the incumbent. An additional read-only handoff audit finds 96 failed evaluations (32 unique physical failures): every failure moves toward the goal and remains inside the current stage's measured support, but the existing transition rule demands an unavailable prospective successor. These exclusions therefore do not establish that the field support itself has ended. A continuation term does not repair that stage-availability rule. See `route_decision/HANDOFF_FAILURES.md`; its diagnostic work is separate.

Candidate generation uses the production incumbent connector as a graph-search seed. Each subsequent controller/terminal evaluation can rejoin its route suffix through any production-eligible connector. It does not enforce a common connector or persistent branch commitment. Independent review records skipped seeds and realized suffix/prefix comparisons; see `route_decision/INDEPENDENT_REVIEW.md`.

The continuation term uses the existing graph quadrature at nominal speed 1 m/s, unchanged source precedence and unknown-risk prior, with no terminal effort estimate. It is approximate and is not an HJB value or a certificate about unseen terrain.

## Reconstruction

On the exact original 24 fixed-branch query sets, global corrected fitting improves/worsens supported value MAE in 12/12 cases. Compact fitting improves 14, worsens 7 and leaves 3 unchanged; mean MAE change versus local-only fitting is −7.53e-5. The prior local-only errors and supported counts reproduce to 1e-12.

On the older 160 full-graph snapshots, global corrected fitting improves/worsens in 129/31; compact fitting in 96/61 with 3 unchanged. Compact has 18 validation rejections. Committed versions, weights, corrections and routing snapshots are retained after routine support synchronization; pruning already-evicted support coordinates is housekeeping, not candidate commitment. Compact fitting guarantees locality, not uniformly better estimation or navigation.

## Messages, selection and faults

The matched-evidence benchmark refits the field on exactly the same four real measurements given to raw and sampled-costmap arms: identical coordinates, support and source IDs. Mean bytes change from 4,442.5 to 1,724.6 for field packets, 2,048.8 to 793.8 for raw, and 2,049.5 to 793.8 for sampled costmaps. Controlled query MAE is 0.03131, 0.05204 and 0.05151 respectively. The field fit uses only the selected batch, not privileged full history. These are twelve matched groups, not independent navigation trials.

Fixed-rate link checks perform actual encoding, queued decoding, normal packet validation and ordinary receiver updates with exactly-once delivery. Operational payloads average 2,675.7 bytes; receiver-relevant advertisements add 5,071.8 bytes in those checks. The stage-selector responds to requests, but bandwidth superiority is not established.

Fault audits use 160 original preserved complementary receiver snapshots. The previous fault campaign did not save its complete contexts, so those old contexts were not silently regenerated. Corrected fitting changes biased-case supported MAE by −0.000775 versus withholding, compared with legacy +0.000884. However, 71/160 corrected biased fits are worse than their matched clean counterparts.

The separately frozen four-scale local consistency gate quarantines all biased packets but also 18/160 clean packets, including 14 useful updates. It removes the corrected receiver's mean biased-case improvement and slightly weakens clean-case improvement. This is an uncalibrated opt-in check with an observed tradeoff, not a robust-fusion success. It remains off by default. No locally unobserved bias-identification guarantee is claimed.

The initial consistency attempt performed computation before failing diagnostic scalar serialization. Its 1,280 attempted updates and partial files are preserved and charged separately; the valid rerun changes serialization only. The communication baseline retains its exact frozen source archive after a concurrent overlap edge-case correction; current-source hashes were not substituted.

'''
    if nav.exists():text+='Additional actual fixed-rate navigation comparison: [report](communication/navigation_v2/REPORT.md). All 32 agent missions complete without collisions or timeouts, but receiver-relevant selection provides no mission benefit and adds substantial status bytes. Both selectors use compact transport. The first two pre-existing layouts, four channel directions and two selectors give 16 episodes; disabled-selector pairs match exactly. A stage-selection/ledger interaction bug found in the first pilot is preserved under `communication/navigation/`; the corrected source amendment, 108/108 nonempty eligible controlled pairs and rerun are separate. Its source/configuration and transport accounting are separate from the snapshot and controlled-request results.\n\n'
    else:text+='Actual navigation gains from the receiver-relevant selector remain untested in the completed snapshot/transport evidence above.\n\n'
    text+='''## Physical structure and verification

Smooth blending removes the controlled fixed-state active-stage handoff jump while using unchanged physical integration and protection. Complete potential derivatives include partition-weight derivatives. Replay/execution controls agree, and logged landscape forces reproduce to 4.44e-16 in the controlled checks. This holds within fixed supported membership. Stage creation/eviction, hard support boundaries, model updates and target changes remain context events. Spatial blending does not inherit the original per-stage radial-progress bound automatically; general switched-system stability and passivity remain unproved.

The production paths are tested together: compact receiver fitting, smooth potential blending, route decisions and receiver-summary roundtrips. Independent route audits recompute terminal quadrature and check update caps, measurement isolation and matching source/input hashes. Three route GIFs use 60 synchronized logged frames each; all exclusions and failures remain in component inventories.

The full test invocation must run from `CoPH-Stagewise/`. An initial repository-root invocation generated missing relative-artifact errors and is preserved separately. The valid full run still reports the existing archived-source-hash failure; its old archive is not rewritten. Native benchmark tests pass separately. See `tests.log`, `native_tests.log` and `verification.json`.

'''
    text+='Full test result: `'+checks['test_result_tail'][0]+'`.\n\n' if checks['test_result_tail'] else ''
    text+='''Commands and opt-in interfaces: [MECHANISM_REPAIRS.md](../../MECHANISM_REPAIRS.md). Detailed records: [route decisions](route_decision/REPORT.md), [reconstruction](reconstruction/REPORT.md), [exact old queries](reconstruction/ORIGINAL_QUERY_REPORT.md), [communication](communication/FINDINGS.md), [overlap](overlap/final_v2/REPORT.md).

The native assignment/static-target development success remains a separate finding. These changes do not turn model revision into demonstrated receiver-learning or historical-replay navigation superiority. Further experiments must preserve the support, information and comparison contracts rather than tune them to force a favorable result.
'''
    (out/'REPORT.md').write_text(text)
    print(json.dumps(dict(route_episodes=24,route_goals=48,route_gifs=3,tests=checks['test_result_tail'])))

if __name__=='__main__':main()
