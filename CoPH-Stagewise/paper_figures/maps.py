"""Drawing the surveyed corridors, hidden doors and logged paths from a scene spec."""
import numpy as np
from matplotlib.patches import Rectangle,Circle,Polygon,FancyArrowPatch
from .style import WALL,FLOOR,DOOR,HIDDEN,AGENT,MESSAGE,fs


def corridors(ax,spec,pad=1.15):
    nodes=np.asarray(spec['network_nodes'],float);r=spec['corridor_radius'];ax.set_facecolor(WALL)
    for i,j in spec['network_edges']:
        a,b=nodes[i],nodes[j];d=b-a;L=np.linalg.norm(d);n=np.array([-d[1],d[0]])/L*r
        ax.add_patch(Polygon([a+n,b+n,b-n,a-n],closed=True,facecolor=FLOOR,edgecolor='none',zorder=0))
        for z in (a,b):ax.add_patch(Circle(z,r,facecolor=FLOOR,edgecolor='none',zorder=0))
    lo,hi=nodes.min(0)-r-pad,nodes.max(0)+r+pad
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]));ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for side in ax.spines.values():side.set_visible(False)


def doors(ax,spec,known=None,label=True):
    """Doors as their exact slab footprint. `known` maps a door to a short label."""
    for o in spec['obstacles']:
        if len(o)!=5:continue
        c=np.asarray(o[:2]);h=np.asarray(o[3:5]);u=h/np.linalg.norm(h);n=np.array([-u[1],u[0]])*o[2];e=h+u*o[2]
        seen=known is None or tuple(o) in known
        ax.add_patch(Polygon([c-e+n,c+e+n,c+e-n,c-e-n],closed=True,facecolor=DOOR if seen else HIDDEN,edgecolor='black',lw=.4,zorder=4))


def path(ax,samples,color,lw=1.6,until=None,**kw):
    p=np.array([s['x'][:2] for s in samples if until is None or s['time']<=until])
    if len(p):ax.plot(*p.T,color=color,lw=lw,solid_capstyle='round',**kw)
    return p


def callout(ax,q,number,color,offset=(0,0)):
    q=np.asarray(q)+offset
    ax.add_patch(Circle(q,.55,facecolor='white',edgecolor=color,lw=.9,zorder=12))
    ax.text(*q,str(number),ha='center',va='center',fontsize=fs(6.5),weight='bold',color=color,zorder=13)


def message(ax,start,end,rad=-.3,lw=1.2):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=fs(7),color=MESSAGE,lw=lw,connectionstyle=f'arc3,rad={rad}',zorder=11,shrinkA=2,shrinkB=2))


def endpoints(ax,spec):
    """Start markers for each agent; each agent's goal is drawn as a star."""
    for i,(s,g) in enumerate(zip(spec['initials'],spec['goals'])):
        ax.scatter(*s[:2],marker='o',s=fs(14),color=AGENT[i],edgecolor='white',lw=.5,zorder=9)
        ax.scatter(*g[:2],marker='*',s=fs(34),color=AGENT[i],edgecolor='white',lw=.4,zorder=9)
