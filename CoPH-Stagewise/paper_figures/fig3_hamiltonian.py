"""F3: a warning reshapes the receiver's Hamiltonian; physical energy is untouched.

Panels (a, b) evaluate A's exact logged control-time model one step before and at
the first warning. Nothing is reconstructed or interpolated between model versions.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from coph_stagewise.records import Config
from coph_stagewise.maze_visuals import landscape
from coph_stagewise.maze_geometry import distance_to_network
from .style import setup_style,save_figure,fs,AGENT,MESSAGE,DOOR,ROOT
from .data import results
from . import maps

DEFAULT=ROOT/'results'/'crossing_traverses_v1'
RECEIVER=0
VMAX=3.


def moment(e,i):
    """Records just before and at the first warning received by agent i."""
    t=min(g['time'] for g in e['geometry_events'][i] if g['kind']=='received_geometry')
    rows=[r for r in e['trajectories'][i] if 'control_model' in r]
    before=max((r for r in rows if r['time']<t-1e-9),key=lambda r:r['time']);after=min((r for r in rows if r['time']>=t-1e-9),key=lambda r:r['time'])
    return t,before,after


def slice_panel(ax,e,row,cfg,window,snapshot,title):
    spec=e['spec'];maps.corridors(ax,spec,pad=0)
    xs=np.linspace(*window[0],150);ys=np.linspace(*window[1],70);xx,yy=np.meshgrid(xs,ys);pts=np.c_[xx.ravel(),yy.ravel()]
    H,F,support=landscape(row['control_model'],pts,np.asarray(row['x'][2:]),cfg)
    inside=distance_to_network(pts,spec['network_nodes'],spec['network_edges'])<=spec['corridor_radius']
    image=ax.pcolormesh(xx,yy,np.ma.masked_where(~inside.reshape(xx.shape),np.clip(H,0,VMAX).reshape(xx.shape)),cmap='viridis',vmin=0,vmax=VMAX,shading='auto',zorder=1,rasterized=True)
    ax.contour(xx,yy,np.ma.masked_where(~inside.reshape(xx.shape),H.reshape(xx.shape)),levels=[.25,.5,1,1.5,2,2.5],colors='white',linewidths=.35,alpha=.6,zorder=2)
    for s in snapshot['stages']:
        x0,x1,y0,y1=s['bounds'];ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,ec='white',lw=.6,ls=(0,(3,2)),zorder=3))
        for states in s['counterfactuals']:
            c=np.array(states)[:,:2];ax.plot(*c.T,color='#FFD54F',lw=1.4,zorder=4)
    q=np.asarray(row['x'][:2]);ax.scatter(*q,s=fs(16),color=AGENT[RECEIVER],edgecolor='white',lw=.6,zorder=8)
    ax.scatter(*row['waypoint'],marker='X',s=fs(18),color='white',edgecolor='black',lw=.4,zorder=8)
    for v,color in ((row['attraction'],'#FF5252'),(row['u'],'black')):
        ax.annotate('',xy=q+np.asarray(v),xytext=q,arrowprops=dict(arrowstyle='-|>',color=color,lw=1.,mutation_scale=fs(6),shrinkA=0,shrinkB=0),zorder=9)
    ax.set(xlim=window[0],ylim=window[1]);ax.set_title(title,fontsize=fs(6.8),pad=2,linespacing=1.15)
    return image


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=DEFAULT);ap.add_argument('--seed',type=int,default=76000);args=ap.parse_args(argv)
    episodes,sources=results(args.results);setup_style(5.5)
    on,off=episodes[(args.seed,'reciprocal_fixed')],episodes[(args.seed,'disabled')];cfg=Config(**on['config'])
    t,before,after=moment(on,RECEIVER)
    snap=min(on['snapshots'][RECEIVER],key=lambda s:abs(s['time']-t))
    q=np.asarray(after['x'][:2]);window=((q[0]-1.9,q[0]+3.3),(q[1]-1.25,q[1]+1.25))
    fig=plt.figure(figsize=(5.5,2.25))
    gs=fig.add_gridspec(2,3,width_ratios=[1,1,.98],height_ratios=[1,1],left=.015,right=.985,top=.86,bottom=.27,wspace=.3,hspace=.14)
    a=fig.add_subplot(gs[:,0]);b=fig.add_subplot(gs[:,1])
    image=slice_panel(a,on,before,cfg,window,snap,f'A at {before["time"]:.2f} s\nbefore the warning')
    slice_panel(b,on,after,cfg,window,snap,f'A at {after["time"]:.2f} s\nwarning just received')
    a.text(window[0][1]-.1,window[1][0]+.15,'door 8 m ahead →',ha='right',va='bottom',fontsize=fs(6),color='white',zorder=10)
    b.text(window[0][1]-.1,window[1][0]+.15,'door known via B',ha='right',va='bottom',fontsize=fs(6),color='white',zorder=10)
    cax=fig.add_axes([a.get_position().x0+.02,.205,b.get_position().x1-a.get_position().x0-.04,.025])
    fig.colorbar(image,cax=cax,orientation='horizontal').set_label('H(q, p) at the logged momentum (J)',labelpad=1)
    cax.tick_params(length=2,pad=1)
    force=fig.add_subplot(gs[0,2]);where=fig.add_subplot(gs[1,2],sharex=force)
    turn=max(r['x'][0] for r in off['trajectories'][RECEIVER] if abs(r['x'][1]-4)<.5 and r['x'][0]<11);end=32.
    for e,ls,label in ((off,(0,(2,1.5)),'no communication'),(on,'-','reciprocal')):
        rows=[r for r in e['trajectories'][RECEIVER] if r['time']<=end]
        force.plot([r['time'] for r in rows],[r['u'][0] for r in rows],color=AGENT[RECEIVER],ls=ls,lw=1.,label=label)
        # Only while in the warned row; elsewhere x is not depth into the dead end.
        where.plot([r['time'] for r in rows],[r['x'][0] if abs(r['x'][1]-after['x'][1])<.5 else np.nan for r in rows],color=AGENT[RECEIVER],ls=ls,lw=1.1,label=label)
    door=[o for o in on['spec']['obstacles'] if len(o)==5 and abs(o[1]-after['x'][1])<.5][0][0]
    where.axhline(door,color=DOOR,lw=1.2);where.text(end-.5,door-.35,'door',ha='right',va='top',fontsize=fs(6),color=DOOR)
    for ax in (force,where):ax.axvline(t,color=MESSAGE,lw=.6,ls=(0,(2,2)));ax.grid(color='#E8EAED',lw=.5)
    force.text(t+.6,.55,'warning',color=MESSAGE,fontsize=fs(6),va='bottom')
    force.axhline(0,color='#9AA1A9',lw=.5);force.set_ylabel('force (N)',labelpad=1)

    force.tick_params(labelbottom=False);force.set_title('Receiver A',fontsize=fs(6.8),pad=2)
    where.legend(fontsize=fs(6),loc=(.02,.36),handlelength=1.6,borderaxespad=0);where.set_ylabel('depth (m)',labelpad=1);fig.align_ylabels([force,where]);where.set_xlabel('time (s)',labelpad=1);where.set_xlim(0,end);where.set_ylim(-.5,door+.8)
    where.annotate(f'turns at {turn:.1f} m',(min(r['time'] for r in off['trajectories'][RECEIVER] if r['x'][0]>=turn-1e-9),turn),xytext=(31.5,3.),ha='right',fontsize=fs(6),
                   arrowprops=dict(arrowstyle='-',lw=.5,color='#555'),color='#333')
    residual=max(abs(r['residual']) for e in (on,off) for rr in e['trajectories'] for r in rr)
    for ax,label,x in ((a,'a',0),(b,'b',0),(force,'c',-.3)):ax.text(x,1.02,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom',ha='left')
    handles=[Line2D([],[],color='#FF5252',label='attraction to waypoint'),Line2D([],[],color='black',label='applied control'),
             Line2D([],[],marker='X',color='white',markeredgecolor='black',lw=0,ms=4,label='route waypoint'),
             Line2D([],[],color='white',ls=(0,(3,2)),lw=.6,label='retained stage windows'),Line2D([],[],color='#FFD54F',label='counterfactual replays')]
    leg=fig.legend(handles=handles,loc='lower center',ncol=5,fontsize=fs(6),handlelength=1.4,columnspacing=1.,bbox_to_anchor=(.5,-.015))
    leg.legend_handles[3].set_color('#555')
    save_figure(fig,'fig3_hamiltonian',sources,notes=dict(seed=args.seed,receiver='A',warning_time=t,before=before['time'],after=after['time'],max_physical_energy_residual_J=residual))
    plt.close(fig)


if __name__=='__main__':main()
