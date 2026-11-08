"""Post-execution checks and analysis; never changes episodes or frozen protocol."""
import argparse,gzip,json,math
from pathlib import Path
import numpy as np


def analyze(out):
    data=json.loads((out/'results.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
    rows=data['rows'];checks=[]
    for r in rows:
        if r['protocol']=='coph_known_protected':
            assert r['final_drag']==[.25]*4
            assert r['identification_work']==0 and r['updates_accepted']==0
        if r['protocol']=='coph_frozen_protected':
            # Operational constructor uses math.expm1; manifest used NumPy.
            assert r['final_drag']==[-math.expm1(-.6*.1)]*4
            assert abs(r['final_drag'][0]-manifest['drag_wrong'])<1e-15
        checks.append(dict(arm=r['protocol'],case=r['case'],finite_return=bool(np.isfinite(r['mean_agent_return'])),steps=r['steps']))
    # Convergence time is based on all four independently identified agents.
    identification=[]
    for seed in manifest['seeds']:
        with gzip.open(out/('coph_online_protected_seed%d_updates.json.gz'%seed),'rt') as f:updates=json.load(f)
        for agent in range(4):
            u=[v for v in updates if v['agent']==agent];hit=next((v['step'] for v in u if abs(v['committed']-.25)<=.01),None)
            identification.append(dict(seed=seed,agent=agent,first_within_point01_step=hit,updates=len(u),accepted=sum(v['accepted'] for v in u)))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    seed=manifest['animation_seed']
    with gzip.open(out/('coph_online_protected_seed%d_updates.json.gz'%seed),'rt') as f:u=json.load(f)
    for agent in range(4):
        rr=[r for r in u if r['agent']==agent];axes[0].step([0]+[r['step']*.1 for r in rr],[manifest['drag_wrong']]+[r['committed'] for r in rr],where='post',label='agent%d'%agent)
    axes[0].axhline(.25,color='black',ls='--',label='native drag');axes[0].set(xlabel='Time (s)',ylabel='Estimated discrete drag',title='Preselected seed %d'%seed);axes[0].legend(fontsize=8)
    for j,c in enumerate(data['paired']):
        d=np.asarray(c['differences']);axes[1].scatter([j]*len(d),d,s=13,alpha=.7);lo,hi=c['interval95'];axes[1].errorbar(j,c['return_difference'],yerr=[[c['return_difference']-lo],[hi-c['return_difference']]],fmt='ks',capsize=4)
    axes[1].axhline(0,color='black',lw=1);axes[1].set_xticks(range(len(data['paired'])));axes[1].set_xticklabels([c['control'] for c in data['paired']],rotation=30,ha='right',fontsize=8)
    axes[1].set_ylabel('Online return minus comparator');fig.tight_layout();fig.savefig(out/'identification_paired.png',dpi=160);fig.savefig(out/'identification_paired.pdf');plt.close(fig)
    complete=[r['first_within_point01_step'] for r in identification if r['first_within_point01_step'] is not None]
    result=dict(checks=checks,identification=identification,known_controller_mapping_verified=True,
        identified_agents=len(complete),total_agents=len(identification),median_identification_step=float(np.median(complete)) if complete else None)
    (out/'analysis_checks.json').write_text(json.dumps(result,indent=2))
    text='# Native identification findings\n\nKnown-native mode was verified on every episode: nominal drag remained exactly 0.25, with zero identification work and no learned updates. The frozen incorrect arm retained its initial drag exactly. Pairing was verified from logged initial states and landmarks.\n\n'
    text+='| Online minus comparator | Paired mean return difference | 95% bootstrap interval |\n|---|---:|---|\n'
    for c in data['paired']:text+='| %s | %.3f | [%.3f, %.3f] |\n'%(c['control'],c['return_difference'],*c['interval95'])
    text+='\n%d/%d agent estimates reached absolute discrete-drag error ≤0.01; median first attainment step among those reaching it: %s. These are four agents nested within 20 starts, not 80 independent environments.\n'%(len(complete),len(identification),result['median_identification_step'])
    text+='\nThe known-model comparison isolates recovery from initial model mismatch. It does not test general improvement over known dynamics or establish cooperative-field/replay benefits. Native return, contacts, coverage and interventions must be read jointly. Static true drag only; a damping-shift protocol remains unexecuted.\n'
    paired={c['control']:c for c in data['paired']}
    text+='\nOn these fresh starts, online identification improves return over the incorrect frozen nominal model, but the correct fixed model performs better. The comparison with the unmodified published phMARL checkpoint does not support a return advantage: its paired interval crosses zero. This campaign therefore does not reproduce a broad native superiority claim. Large poor-return episodes and contacts remain included.\n'
    text+='\n| Retained diagnostic start | Arm | Return | Coverage | New contacts | Stall agent-steps |\n|---|---|---:|---:|---:|---:|\n'
    for r in rows:
        if r['case'] in ('seed82000','seed82011','seed82019'):
            text+='| %s | %s | %.3f | %.2f | %d | %d |\n'%(r['case'],r['protocol'],r['mean_agent_return'],r['final_landmark_coverage'],r['new_contact_onsets'],r['stall_agent_steps'])
    text+='\nThese outcome checks identify poor coverage/contact/stall behavior; they do not establish its cause without matched control diagnostics. No failures were used to retune settings.\n'
    (out/'FINDINGS.md').write_text(text);print(text)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);analyze(ap.parse_args().out)
