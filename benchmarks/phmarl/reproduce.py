#!/usr/bin/env python3
"""Pinned published-checkpoint evaluation; never modifies the upstream checkout."""
import os
os.environ.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MPLBACKEND='Agg')
import argparse
import contextlib
import hashlib
import importlib.util
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
sys.dont_write_bytecode = True
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from vmas import make_env

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = Path(os.environ.get('COPH_PHMARL_ROOT', ROOT / 'external/phMARL')).expanduser().resolve()
CHECKPOINT = UPSTREAM / 'data/actor_net_simple_spread_1400000_LEMURS.pth'
COMMIT = 'b529c05b7ec29c26d55627657131e26ef39ff3b1'

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def cpu(actor):
    actor.to('cpu')
    for module in actor.modules():
        if hasattr(module, 'device'):
            module.device = torch.device('cpu')
    actor.na = 4
    actor.eval()
    return actor

def position(env):
    return np.stack([x.state.pos[0].numpy().copy() for x in env.world.agents])

def landmarks(env):
    return np.stack([x.state.pos[0].numpy().copy() for x in env.world.landmarks])

def record(env):
    return {'positions': [position(env)], 'landmarks': landmarks(env), 'actions': [], 'rewards': [], 'collisions': []}

def append(rec, env, actions, rewards):
    rec['positions'].append(position(env))
    rec['actions'].append(torch.stack(actions).detach().numpy().copy())
    rec['rewards'].append(torch.stack(rewards).numpy().copy())
    rec['collisions'].append(sum(int(env.world.is_overlapping(a, b)[0])
        for i, a in enumerate(env.world.agents) for b in env.world.agents[i+1:]))

def metrics(rec, protocol, episode):
    d = np.linalg.norm(rec['positions'][-1][:,None] - rec['landmarks'][None,:], axis=-1)
    i, j = linear_sum_assignment(d)
    rewards = np.asarray(rec['rewards']).reshape(-1, 4)
    return dict(protocol=protocol, episode=episode, steps=len(rewards),
        summed_agent_return=float(sum(float(x.sum()) for x in rewards)),
        mean_agent_return=float(sum(float(x.mean()) for x in rewards)),
        final_assignment_distance=float(d[i,j].mean()),
        final_landmark_coverage=float((d.min(axis=0)<=.15).mean()),
        collision_pair_steps=int(sum(rec['collisions'])),
        collision_free=not any(rec['collisions']))

def native():
    """Execute original main with only CPU portability and rendering interception."""
    spec = importlib.util.spec_from_file_location('upstream_evaluation', UPSTREAM/'evaluation.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    histories = []
    real_load = torch.load
    old_cuda = torch.has_cuda
    cwd = Path.cwd()
    def factory(**kw):
        env = make_env(**kw)
        reset, step = env.reset, env.step
        def reset_record(*a, **k):
            obs = reset(*a, **k)
            histories.append(record(env))
            return obs
        def step_record(actions):
            result = step(actions)
            append(histories[-1], env, actions, result[1])
            return result
        env.reset, env.step = reset_record, step_record
        # Rendering is observational only; logged trajectories are rendered later.
        env.render = lambda **kw: np.zeros((1,1,3), dtype=np.uint8)
        return env
    args = argparse.Namespace(scenario_name='simple_spread', seed=1, max_frames_eval=400,
        num_agents_eval=4, n_agents_good_eval=4, n_agents_adversaries_eval=4,
        ratio_eval=2, r_communication=.45, batch_size=1024, num_envs=96,
        ratio=4., reward_scaling=1, neural_network_name='ph-MARL',
        desired_frame=1400000, num_test_episodes=10)
    try:
        module.make_env = factory
        torch.has_cuda = False
        torch.load = lambda *a, **k: cpu(real_load(*a, map_location='cpu', **k))
        os.chdir(UPSTREAM)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module.main(args)
    finally:
        os.chdir(cwd)
        torch.load = real_load
        torch.has_cuda = old_cuda
    return histories, output.getvalue()

def wrapper(steps, random=False):
    torch.manual_seed(1)
    np.random.seed(1)
    env = make_env(scenario_name='simple_spread', num_envs=1, device='cpu',
        continuous_actions=True, wrapper=None, seed=1, max_steps=400, n_agents=4,
        n_agents_good=4, n_agents_adversaries=4, n_packages=1, share_reward=True, ratio=2)
    actor = cpu(torch.load(CHECKPOINT, map_location='cpu'))
    generator = torch.Generator().manual_seed(100001)
    histories = []
    for episode in range(10):
        obs = env.reset(seed=None)
        rec = record(env)
        for _ in range(steps):
            with torch.no_grad():
                action = torch.empty((1,4,2)).uniform_(-1,1,generator=generator) if random else actor(torch.stack(obs).squeeze(1).unsqueeze(0)).mean
            actions = list(action.transpose(0,1))
            obs, rew, done, _ = env.step(actions)
            append(rec, env, actions, rew)
            if bool(torch.as_tensor(done).any()):
                break
        histories.append(rec)
    return histories

def render(out, rec):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.patches import Circle
    p = np.asarray(rec['positions'])
    fig, ax = plt.subplots(figsize=(6,6))
    lim = max(1.3, float(np.abs(p).max())+.2)
    ax.set(xlim=(-lim,lim), ylim=(-lim,lim), aspect='equal', xlabel='x', ylabel='y')
    ax.scatter(*rec['landmarks'].T, marker='x', s=100, c='black', label='Landmarks')
    colors = ['tab:blue','tab:orange','tab:green','tab:red']
    circles = [Circle(p[0,i], .15, color=colors[i], alpha=.6) for i in range(4)]
    lines = [ax.plot([],[],color=colors[i],lw=1)[0] for i in range(4)]
    for c in circles: ax.add_patch(c)
    title = ax.set_title('')
    def update(t):
        for i in range(4):
            circles[i].center = p[t,i]
            lines[i].set_data(p[:t+1,i,0],p[:t+1,i,1])
        title.set_text('Published phMARL: navigation, episode 0\nSimulator step %d / 400' % t)
        return circles+lines+[title]
    FuncAnimation(fig,update,frames=list(range(0,len(p),4)),interval=80).save(out/'navigation_episode0.gif',writer=PillowWriter(fps=12))
    update(len(p)-1)
    fig.savefig(out/'navigation_episode0.png',dpi=160)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results/native_v1')
    args = parser.parse_args()
    if args.out.exists(): raise SystemExit('Output already exists; use a new directory.')
    args.out.mkdir(parents=True)
    sys.path.insert(0,str(UPSTREAM))
    import functions
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    commit = subprocess.check_output(['git','-C',str(UPSTREAM),'rev-parse','HEAD'],text=True).strip()
    assert commit == COMMIT
    assert not subprocess.check_output(['git','-C',str(UPSTREAM),'status','--porcelain','--untracked-files=no'],text=True).strip()
    import vmas
    scenario_path = Path(vmas.__file__).parent / "scenarios/mpe/simple_spread.py"
    manifest = dict(commit=commit,checkpoint=str(CHECKPOINT.relative_to(ROOT)),checkpoint_sha256=digest(CHECKPOINT),
        evaluator_sha256=digest(UPSTREAM/'evaluation.py'),runner_sha256=digest(__file__),
        scenario_sha256=digest(scenario_path),python=sys.version, executable=sys.executable,
        packages={k:importlib.metadata.version(k) for k in ['torch','numpy','vmas','gym','scipy','Pillow','matplotlib']},
        seed=1,episodes=10,agents=4,device='cpu',threads=1,checkpoint_frame=1400000,
        protocols={'upstream_original':200,'wrapper_legacy_check':200,'corrected_400':400,'random_400':400},
        reset='one seeded environment; reset(seed=None) each episode, as upstream',
        animation_episode=0, adaptations=['CPU tensor and device-attribute remapping','render interception only in original evaluator'],
        platform=platform.platform())
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (args.out/'runner.py.txt').write_text(Path(__file__).read_text())
    start=time.perf_counter()
    histories, stdout = native()
    (args.out/'upstream_stdout.txt').write_text(stdout)
    all_runs={'upstream_original':histories}
    print('Original evaluator complete',flush=True)
    for name, steps, random in [('wrapper_legacy_check',200,False),('corrected_400',400,False),('random_400',400,True)]:
        all_runs[name]=wrapper(steps,random)
        print(name+' complete',flush=True)
    parity={}
    for key in ['positions','landmarks','actions','rewards','collisions']:
        parity[key]=all(np.array_equal(a[key],b[key]) for a,b in zip(histories,all_runs['wrapper_legacy_check']))
    assert all(parity.values()), parity
    pairing=all(np.array_equal(a['positions'][0],b['positions'][0]) and np.array_equal(a['landmarks'],b['landmarks'])
        for a,b in zip(all_runs['corrected_400'],all_runs['random_400']))
    assert pairing
    rows=[]
    for name, records in all_runs.items():
        for i, rec in enumerate(records):
            rows.append(metrics(rec,name,i))
            np.savez_compressed(args.out/('%s_%02d.npz'%(name,i)),**{k:np.asarray(v) for k,v in rec.items()})
    summary={}
    for name in all_runs:
        rr=[r for r in rows if r['protocol']==name]
        summary[name]={k:{'mean':float(np.mean([r[k] for r in rr])), 'std':float(np.std([r[k] for r in rr]))}
            for k in rr[0] if k not in ['protocol','episode']}
    (args.out/'metrics.json').write_text(json.dumps(dict(rows=rows,summary=summary),indent=2))
    (args.out/'checks.json').write_text(json.dumps(dict(original_wrapper_bitwise_parity=parity,
        random_control_initial_state_pairing=pairing,elapsed_seconds=time.perf_counter()-start),indent=2))
    render(args.out,all_runs['corrected_400'][0])
    table='| Evaluation | Actual steps | Mean-agent return | Final assignment distance | Landmark coverage ≤0.15 | Collision pair-steps |\n|---|---:|---:|---:|---:|---:|\n'
    for name in ['upstream_original','corrected_400','random_400']:
        s=summary[name]
        table+='| %s | %.0f | %.2f ± %.2f | %.3f | %.1f%% | %.1f |\n'%(name,s['steps']['mean'],s['mean_agent_return']['mean'],s['mean_agent_return']['std'],s['final_assignment_distance']['mean'],100*s['final_landmark_coverage']['mean'],s['collision_pair_steps']['mean'])
    report='''# Native phMARL navigation checkpoint reproduction

Fresh evaluation of the authors' published checkpoint, without retraining. This reproduces checkpoint execution and evaluator behavior, not the paper's complete training study or statistical claims.

Pinned upstream: `%s`. Checkpoint: `actor_net_simple_spread_1400000_LEMURS.pth`. Native VMAS 1.2.6 `simple_spread`, four agents, ten sequential resets from seed 1, deterministic distribution-mean actions. Package versions and hashes are in manifest.json. The upstream checkout was verified clean and its revision matched remote HEAD before execution.

%s
Values after ± are population standard deviations over ten episodes from one checkpoint. Original and corrected horizons must not be compared as equal-budget returns. The original summed-agent metric is four times the mean-agent return; both are saved in metrics.json. The original stdout is preserved separately.

## Verification and compatibility

The original evaluation.py main was executed with CPU tensor remapping, legacy device attributes changed to CPU, and rendering intercepted. No actor weights, preprocessing, environment dynamics, rewards, or decisions were changed. The original double timestep increment was preserved: a requested 400-step horizon executes 200 simulator steps. An independent wrapper matched original positions, landmarks, actions, rewards and collision counts bitwise in all ten episodes. The corrected wrapper runs 400 actual steps; the random control uses identical initial states and a separate action RNG. Animation episode 0 was selected before execution, not selected for favorable performance.

## Interpretation and comparison contract

Navigation is landmark coverage with agent-agent collision penalties, not our surveyed hazardous corridor task. Native observations include own position/velocity and relative positions of all landmarks and other agents; the checkpoint applies its own preprocessing and communication mask. There is no environmental risk sensor or surveyed graph. Coverage at radius 0.15 and assignment distance are supplementary evaluator metrics, not a native success termination condition. Collision pair-steps count unordered overlapping pairs after each simulator step, including initially crowded layouts; they are not distinct collision events. Do not claim collision-free navigation from high final coverage.

A fair CoPH comparison must use this simulator, its action handling, observations, reward, reset distribution and horizon. Our exact damped-flow plant, risk sensors, graph prior, and collision protection cannot simply be substituted. Inter-agent coordination and target allocation require explicit adaptation. No CoPH-versus-phMARL superiority has been tested here.

## Artifacts

- `metrics.json`: episode rows and summaries, including summed rewards.
- `checks.json`: bitwise original-wrapper parity and matched initial-state checks.
- `*_NN.npz`: positions including initial state, landmarks, actions, rewards and collision counts.
- `navigation_episode0.gif` and `.png`: corrected-horizon logged trajectory, all agents shown at physical radius 0.15.
- `manifest.json`, `runner.py.txt`, `upstream_stdout.txt`: provenance and reproduction evidence.

The existing legacy reproduction files and all CoPH-Stagewise experiments remain unchanged.
'''%(commit,table)
    (args.out/'REPORT.md').write_text(report)
    print(table,flush=True)

if __name__=='__main__': main()
