"""F1: the same two agents, same sensors, with and without reciprocal warnings."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .style import setup_style,save_figure,panel_label,fs,AGENT,MESSAGE,DOOR,ARMS,ARM_SHORT,ARM_COLOR,ROOT
from .data import results,wasted,door_events,oracle,path_length
from . import maps

DEFAULT=ROOT/'results'/'crossing_traverses_v1'


def events_in_order(e):
    """Door events of a run as (time, agent, kind, q, door), in time order."""
    return sorted(((t,i,kind,q,door) for (door,i,kind),(t,q,_) in door_events(e).items()),key=lambda z:z[0])


def draw(episodes,seed,ax_off,ax_on,ax_bar):
    off,on=episodes[(seed,'disabled')],episodes[(seed,'reciprocal_fixed')]
    for ax,e,label in ((ax_off,off,'No communication'),(ax_on,on,'Reciprocal warnings')):
        maps.corridors(ax,e['spec'],pad=.25);maps.doors(ax,e['spec'])
        if e is on:
            for i in (0,1):maps.path(ax,off['samples'][i],AGENT[i],lw=.9,ls=(0,(1.5,1.5)),alpha=.45,zorder=3)
        for i in (0,1):maps.path(ax,e['samples'][i],AGENT[i],zorder=5)
        maps.endpoints(ax,e['spec'])
        team=sum(path_length(s) for s in e['samples']);finish=max(a['finish_time'] for a in e['agents'])
        ax.set_title(f'{label}\nteam path {team:.0f} m · done at {finish:.0f} s',fontsize=fs(6.6),pad=2,linespacing=1.15)
    ordered=events_in_order(on);numbered=[]
    for k,(t,i,kind,q,door) in enumerate(ordered,1):
        # Markers sit in the wall just below the corridor point they label.
        maps.callout(ax_on,q,k,AGENT[i],offset=(0,-1.35))
        numbered.append((k,t,i,kind))
    for (t,i,kind,q,door) in ordered:
        if kind!='received':continue
        src=[z for z in ordered if z[4]==door and z[2]=='sensed' and z[1]!=i]
        if src:maps.message(ax_on,src[0][3],q,rad=.25 if q[0]<src[0][3][0] else -.25)
    items=[f'({k}) {t:.1f} s {"AB"[i]} {"sees a door" if kind=="sensed" else "warned"}' for k,t,i,kind in numbered]
    ax_on.text(.5,-.035,'   '.join(items[:2])+'\n'+'   '.join(items[2:]),transform=ax_on.transAxes,ha='center',va='top',fontsize=fs(6.3),linespacing=1.25)
    x=np.arange(len(ARMS));pred=[oracle(off['spec'],a)['team_wasted'] for a in ARMS]
    for k,arm in enumerate(ARMS):
        values=[sum(wasted(e)) for (s,a),e in episodes.items() if a==arm]
        ax_bar.bar(k,np.mean(values),.66,color=ARM_COLOR[arm],zorder=2)
        ax_bar.scatter([k]*len(values),values,s=fs(4),color='black',zorder=3,lw=0)
        ax_bar.plot([k-.36,k+.36],[pred[k]]*2,color='black',lw=1.,ls=(0,(2,1)),zorder=4)
    ax_bar.set_xticks(x,['none','A→B','B→A','both'],fontsize=fs(6.3))
    ax_bar.set_ylabel('team detour (m)',labelpad=1);ax_bar.set_ylim(0,max(pred)*1.4);ax_bar.set_yticks([0,25,50])
    ax_bar.grid(axis='y',color='#E3E5E8',lw=.5,zorder=0);ax_bar.set_title('Detour by\nwarning channel',fontsize=fs(6.6),pad=2,linespacing=1.15)
    ax_bar.legend(handles=[Line2D([],[],color='black',lw=1.,ls=(0,(2,1)),label='oracle'),
                           Line2D([],[],marker='o',color='black',lw=0,ms=2,label='seeds')],loc='upper right',fontsize=fs(6.3),handlelength=1.4,borderaxespad=0,ncol=2,columnspacing=.6)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=DEFAULT);ap.add_argument('--seed',type=int,default=76000);args=ap.parse_args(argv)
    episodes,sources=results(args.results);setup_style(5.5)
    fig=plt.figure(figsize=(5.5,2.2));gs=fig.add_gridspec(1,3,width_ratios=[1,1,.62],left=.01,right=.99,top=.86,bottom=.26,wspace=.08)
    axes=[fig.add_subplot(gs[k]) for k in range(3)];draw(episodes,args.seed,*axes)
    box=axes[2].get_position();axes[2].set_position([box.x0+.055,box.y0+.03,box.width-.055,box.height-.03])
    for ax,label,x in zip(axes,'abc',(0,0,-.42)):ax.text(x,1.012,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom',ha='left')
    handles=[Line2D([],[],color=AGENT[0],label='A: west → east'),Line2D([],[],color=AGENT[1],label='B: east → west'),
             Line2D([],[],color='gray',ls=(0,(1.5,1.5)),label='same agent, no communication'),
             Line2D([],[],marker='o',color='gray',lw=0,ms=3,label='start'),Line2D([],[],marker='*',color='gray',lw=0,ms=5,label='goal'),
             Patch(facecolor=DOOR,edgecolor='black',lw=.4,label='hidden door'),Line2D([],[],color=MESSAGE,label='warning')]
    fig.legend(handles=handles,loc='lower center',ncol=4,fontsize=fs(6.3),handlelength=1.5,columnspacing=1.2,bbox_to_anchor=(.5,-.01))
    save_figure(fig,'fig1_teaser',sources,notes=dict(seed=args.seed,detour='path length minus true shortest route (doors known)'))
    plt.close(fig)


if __name__=='__main__':main()
