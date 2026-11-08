"""F2: distance-to-goal over time for every channel; warnings drawn sender → receiver."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .style import setup_style,save_figure,fs,AGENT,MESSAGE,ARMS,ROOT
from .data import results,door_events

DEFAULT=ROOT/'results'/'crossing_traverses_v1'
TITLES=dict(disabled='No communication',one_way_01='A → B only',one_way_10='B → A only',reciprocal_fixed='Reciprocal')


def remaining_at(samples,t):
    times=np.array([s['time'] for s in samples]);k=int(np.clip(np.searchsorted(times,t),0,len(times)-1));return samples[k]['remaining']


def series(samples,grid):
    return np.interp(grid,[z['time'] for z in samples],[z['remaining'] for z in samples],right=0.)


def draw(ax,e,end,ghost=None):
    grid=np.linspace(0,end,600)
    for i in (0,1):
        s=e['samples'][i];y=series(s,grid)
        if ghost is not None:
            g=series(ghost['samples'][i],grid)
            ax.plot(grid,g,color=AGENT[i],lw=.8,ls=(0,(2,1.5)),alpha=.55,zorder=2)
            ax.fill_between(grid,y,g,where=g>y+1e-6,color=AGENT[i],alpha=.13,lw=0,zorder=1)
        ax.plot([z['time'] for z in s],[z['remaining'] for z in s],color=AGENT[i],lw=1.3,zorder=3)
    warnings=0
    for (door,i,kind),(t,q,sender) in sorted(door_events(e).items(),key=lambda z:z[1][0]):
        y=remaining_at(e['samples'][i],t)
        if kind=='sensed':ax.scatter(t,y,marker='*',s=fs(30),color=AGENT[i],edgecolor='black',lw=.35,zorder=6)
        else:
            # Alternate labels below and above the curves so neighbours never collide.
            text=(t+9,y-9) if warnings%2==0 else (t+5,y+6.5);warnings+=1
            ax.annotate(f'{"AB"[i]} warned',(t,y),xytext=text,fontsize=fs(6),color=MESSAGE,ha='left',va='center',zorder=7,
                        arrowprops=dict(arrowstyle='-|>',color=MESSAGE,lw=.8,mutation_scale=fs(5),shrinkA=0,shrinkB=1))
    finish=max(a['finish_time'] for a in e['agents'])
    ax.axvline(finish,color='#B0B5BB',lw=.6,ls=(0,(2,2)),zorder=1)
    ax.set_xlim(0,end);ax.grid(color='#E8EAED',lw=.5,zorder=0)
    return finish


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=DEFAULT);ap.add_argument('--seed',type=int,default=76000);args=ap.parse_args(argv)
    episodes,sources=results(args.results);setup_style(5.5)
    present=[a for a in ARMS if (args.seed,a) in episodes];end=max(a['finish_time'] for a in present for a in episodes[(args.seed,a)]['agents'])*1.04
    fig,axes=plt.subplots(1,len(present),figsize=(5.5,2.05),sharey=True,gridspec_kw=dict(left=.075,right=.995,top=.83,bottom=.3,wspace=.08))
    top=0
    for k,(ax,arm) in enumerate(zip(axes,present)):
        e=episodes[(args.seed,arm)];finish=draw(ax,e,end,None if arm=='disabled' else episodes.get((args.seed,'disabled')))
        top=max(top,max(z['remaining'] for s in e['samples'] for z in s))
        ax.set_title(f'{TITLES[arm]}\nteam done at {finish:.0f} s',fontsize=fs(6.8),pad=2,linespacing=1.15);ax.set_xlabel('time (s)',labelpad=1)
        ax.text(-.02 if k else -.2,1.02,'abcd'[k],transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom',ha='right')
    axes[0].set_ylim(0,top*1.08);axes[0].set_ylabel('distance to goal (m)',labelpad=1)
    handles=[Line2D([],[],color=AGENT[0],label='A'),Line2D([],[],color=AGENT[1],label='B'),
             Line2D([],[],marker='*',color='gray',markeredgecolor='black',markeredgewidth=.35,lw=0,ms=6,label='sees a door'),
             Line2D([],[],color=MESSAGE,label='warning arrives'),
             Line2D([],[],color='gray',lw=.8,ls=(0,(2,1.5)),label='no communication'),
             Patch(facecolor='gray',alpha=.25,label='distance saved')]
    fig.legend(handles=handles,loc='lower center',ncol=6,fontsize=fs(6.3),handlelength=1.5,columnspacing=1.1,bbox_to_anchor=(.53,-.01))
    save_figure(fig,'fig2_cascade',sources,notes=dict(seed=args.seed,distance='true corridor distance to goal with doors known; rises while an agent walks into a dead end'))
    plt.close(fig)


if __name__=='__main__':main()
