"""F10 (appendix): native-task trajectories for one shared seed, and return against contact per episode."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .style import setup_style,save_figure,fs
from .native_data import DEFAULT,ARMS,LABEL,COLOR,load

AGENTS=('#0072B2','#D55E00','#009E73','#CC79A7')
SEED=81000
PAIRS=[(i,j) for i in range(4) for j in range(i+1,4)]  # order of contact_pairs columns


def trajectories(data,sources):
    setup_style(5.5);fig,axes=plt.subplots(2,3,figsize=(5.5,3.9),gridspec_kw=dict(left=.02,right=.98,top=.9,bottom=.02,wspace=.08,hspace=.3))
    for ax,arm in zip(axes.ravel(),ARMS):
        r=next(x for x in data[arm] if x['seed']==SEED);p=r['positions'];m=r['metrics']
        for k,(x,y) in enumerate(r['landmarks']):ax.add_patch(Circle((x,y),.15,facecolor='none',edgecolor='#9AA1A9',lw=.6,ls=(0,(2,1.5))))
        for i in range(4):
            ax.plot(p[:,i,0],p[:,i,1],color=AGENTS[i],lw=1.);ax.scatter(*p[0,i],s=fs(8),color=AGENTS[i],marker='o',edgecolor='white',lw=.3,zorder=5)
            ax.add_patch(Circle(p[-1,i],.15,facecolor=AGENTS[i],alpha=.35,edgecolor=AGENTS[i],lw=.5))
        steps,cols=np.nonzero(r['contacts'])
        for t,c in zip(steps,cols):
            i,j=PAIRS[c];mid=(p[t+1,i]+p[t+1,j])/2;ax.scatter(*mid,marker='x',s=fs(10),color='black',lw=.7,zorder=6)
        ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),xticks=[],yticks=[]);ax.set_aspect('equal')
        ax.set_title(f'{LABEL[arm]}\nR {m["mean_agent_return"]:.0f} · overlap {m["collision_pair_steps"]:.0f} · cov. {100*m["final_landmark_coverage"]:.0f}%',fontsize=fs(6.2),pad=2,linespacing=1.1)
        for side in ax.spines.values():side.set_visible(True);side.set_color('#C9CDD2');side.set_linewidth(.5)
    save_figure(fig,'fig10_native_trajectories',sources,notes=dict(seed=SEED));plt.close(fig)


def tradeoff(data,sources):
    setup_style(5.5);fig,ax=plt.subplots(figsize=(3.4,2.3),gridspec_kw=dict(left=.16,right=.97,top=.92,bottom=.18))
    for arm in ARMS:
        x=np.array([r['metrics']['collision_pair_steps'] for r in data[arm]]);y=np.array([r['metrics']['mean_agent_return'] for r in data[arm]])
        ax.scatter(x+.6,y,s=fs(9),color=COLOR[arm],edgecolor='black',lw=.25,alpha=.85,zorder=3,label=LABEL[arm])
        ax.scatter(x.mean()+.6,y.mean(),s=fs(40),marker='D',color=COLOR[arm],edgecolor='black',lw=.6,zorder=4)
    ax.set_xscale('log');ax.set(xlabel='overlap pair-steps + 0.6 (log)',ylabel='mean-agent return')
    ax.grid(color='#E8EAED',lw=.5);ax.legend(fontsize=fs(5.6),loc='lower left',handletextpad=.2,borderaxespad=.2,ncol=2,columnspacing=.6)
    ax.set_title('Return against contact, 20 episodes per arm (diamonds: means)',fontsize=fs(6.6),pad=2)
    save_figure(fig,'fig10_native_tradeoff',sources);plt.close(fig)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=DEFAULT);args=ap.parse_args(argv)
    data,sources=load(args.results);trajectories(data,sources);tradeoff(data,sources)


if __name__=='__main__':main()
