"""Certify whether a new contact was avoidable by any admissible control (|u_x|,|u_y| <= 1).

Replays native physics (drag .25, dt .1, VMAS penalty contact force between all agents) from the reset state.
The two agents of the contact pair apply full input directly away from each other at every step (bang-bang along
the current line between them, the best escape under per-component bounds); all other agents replay their logged
actions. If the pair still comes within the contact distance by the logged onset step, the contact is certified
unavoidable. The escape is simulated for 10 steps past the onset, so a contact that can only be postponed is
still certified. Uncertified contacts count as avoidable.
"""
import numpy as np
from protection import contact_impulses


def unavoidable(positions,actions,pair,onset,contact=.30,dt=.1,drag=.25,extra=10):
    a,b=pair;q=np.asarray(positions[0],float).copy();v=np.zeros_like(q);best=np.inf
    for t in range(min(onset+1+extra,len(actions))):
        u=np.asarray(actions[t],float).reshape(len(q),2).copy()
        d=q[a]-q[b];n=d/max(np.linalg.norm(d),1e-12)
        u[a]=np.sign(n)*(np.abs(n)>1e-9);u[b]=-u[a]
        v=(1-drag)*v+dt*u+contact_impulses(q,dt=dt);q=q+dt*v
        best=min(best,float(np.linalg.norm(q[a]-q[b])))
    return best<contact,best


def certify(npz):
    import itertools
    pairs=list(itertools.combinations(range(4),2));cp=npz['contact_pairs'];prev=np.vstack([npz['initial_contacts'][None],cp[:-1]])
    out=[]
    for t,j in np.argwhere(cp&~prev):
        ok,best=unavoidable(npz['positions'],npz['actions'],pairs[j],int(t))
        out.append(dict(step=int(t),pair=list(pairs[j]),unavoidable=bool(ok),best_case_distance=best))
    return out
