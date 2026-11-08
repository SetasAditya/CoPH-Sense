"""Additional controlled native rollouts; no settings are changed from frozen suite."""
import json
from pathlib import Path
import numpy as np
import torch
from vmas import make_env
from safety_suite import attraction,contacts
from protection import Protection

def main():
    torch.set_num_threads(1)
    cases={
        'separated_goals':([[-.7,0],[.7,0],[0,2],[2,2]],[[-.3,0],[.3,0],[0,2],[2,2]]),
        'close_goals':([[-.7,0],[.7,0],[0,2],[2,2]],[[-.115,0],[.115,0],[0,2],[2,2]]),
        'initial_overlap':([[-.1,0],[.1,0],[0,2],[2,2]],[[-.5,0],[.5,0],[0,2],[2,2]]),
        'head_on_swap':([[-.7,0],[.7,0],[0,2],[2,2]],[[.7,0],[-.7,0],[0,2],[2,2]])}
    rows=[]
    for name,(starts,goals) in cases.items():
        env=make_env(scenario_name='simple_spread',num_envs=1,device='cpu',continuous_actions=True,n_agents=4,max_steps=200,seed=82)
        env.reset(seed=82)
        for a,p in zip(env.world.agents,starts):a.set_pos(torch.tensor(p,dtype=torch.float32),batch_index=None);a.set_vel(torch.zeros(2),batch_index=None)
        for l,p in zip(env.world.landmarks,goals):l.set_pos(torch.tensor(p,dtype=torch.float32),batch_index=None)
        obs=[env.scenario.observation(a) for a in env.agents];shield=Protection();initial=contacts(env);previous=initial
        overlaps=0;onsets=0;infeasible=0;unknown=0;first_separated=None
        for k in range(200):
            processed=shield.prepare([o.numpy().copy() for o in obs]);u,logs=shield.apply(attraction(processed))
            assert all(bool((x.abs()<=1).all()) for x in u)
            obs,_,_,_=env.step(u);cp=contacts(env);overlaps+=int(cp.sum());onsets+=int((cp&~previous).sum());previous=cp
            if first_separated is None and not cp.any():first_separated=k+1
            infeasible+=sum(not x['feasible'] for x in logs);unknown+=sum(x['unknown_peer_velocities'] for x in logs)
        q=np.stack([a.state.pos[0].numpy() for a in env.world.agents]);g=np.array(goals)
        coverage=float((np.linalg.norm(q[:,None]-g[None,:],axis=-1).min(axis=0)<=.15).mean())
        rows.append(dict(case=name,initial_pairs=int(initial.sum()),overlap_pair_steps=overlaps,new_onsets=onsets,
            first_all_separated_step=first_separated,final_coverage=coverage,infeasible_controls=infeasible,
            unknown_velocity_occurrences=unknown,final_positions=q.tolist(),starts=starts,goals=goals))
    out=Path(__file__).parent/'results/native_safety_v2/controlled_checks.json'
    out.write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
