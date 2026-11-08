"""F7: share the environmental field, not the decision. Carrier excess cost over the best tested arm."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from coph_stagewise.maze_geometry import distance_to_network
from .style import setup_style,save_figure,fs,AGENT,MESSAGE,ROOT
from .pending import placeholder
from . import maps

ORDER=('none','field','raw','no_learning','occupancy_0p1','occupancy_0p25','occupancy_1','decision')
LABEL=dict(none='no communication',field='learned env. field',raw='raw readings',no_learning='env. field, no receiver fit',
           occupancy_0p1='occupancy, τ = 0.1',occupancy_0p25='occupancy, τ = 0.25',occupancy_1='occupancy, τ = 1',decision="sender's decision")
SCENE_LABEL=dict(mild='1 mild',double_mild='2 mild',moderate='moder.',severe='severe')
SCENES=('mild','double_mild','moderate','severe')
PATH_COLOR=dict(field=MESSAGE,occupancy_0p25='#E69F00',decision='#009E73',none='#7A7F86')


def hazard(ax,spec):
    from coph_stagewise.hazard_oracle import true_risk
    xs=np.linspace(-1,13,280);ys=np.linspace(-1,8.5,190);xx,yy=np.meshgrid(xs,ys);pts=np.c_[xx.ravel(),yy.ravel()]
    r=np.array([true_risk(q,[tuple(p) for p in spec['patches']]) for q in pts]);inside=distance_to_network(pts,spec['network_nodes'],spec['network_edges'])<=spec['corridor_radius']
    ax.pcolormesh(xx,yy,np.ma.masked_where(~inside|(r<.04),r).reshape(xx.shape),cmap='Reds',vmin=0,vmax=.45,shading='auto',zorder=1,rasterized=True)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,nargs='*',default=[]);ap.add_argument('--scene',default='double_mild');args=ap.parse_args(argv)
    if not args.results:return placeholder('fig7_terrain','Share the environmental field, not the decision','graded-hazard pilot runs')
    from .hazard_data import results
    episodes,manifests,sources=results(args.results[0]);by={(e['scene'],e['arm']):e for e in episodes}
    setup_style(5.5);fig=plt.figure(figsize=(5.5,2.55))
    a=fig.add_axes([.005,.2,.37,.66]);b=fig.add_axes([.62,.2,.255,.66])
    spec=by[(args.scene,'field')]['spec'];maps.corridors(a,spec,pad=.15);hazard(a,spec)
    scout=np.array(by[(args.scene,'field')]['samples'][0]);a.plot(*scout.T,color='#555',lw=.9,ls=(0,(1.5,1.2)),zorder=4)
    for arm in ('none','decision','occupancy_0p25','field'):
        p=np.array(by[(args.scene,arm)]['samples'][1]);a.plot(*p.T,color=PATH_COLOR[arm],lw=1.5 if arm=='field' else 1.1,zorder=6 if arm=='field' else 5,alpha=1 if arm=='field' else .9)
    a.scatter(0,4,s=fs(14),color='black',zorder=8);a.scatter(12,4,marker='*',s=fs(36),color='black',zorder=8)
    a.set_title(f'Carrier routes, scene "{SCENE_LABEL[args.scene]}"',fontsize=fs(6.8),pad=2)
    a.legend(handles=[Line2D([],[],color=PATH_COLOR['field'],label='env. field'),Line2D([],[],color=PATH_COLOR['occupancy_0p25'],label='occupancy τ=0.25'),
                      Line2D([],[],color=PATH_COLOR['decision'],label='decision'),Line2D([],[],color=PATH_COLOR['none'],label='no comm.'),
                      Line2D([],[],color='#555',ls=(0,(1.5,1.2)),label='scout')],loc='upper center',bbox_to_anchor=(.5,-.01),ncol=3,fontsize=fs(6),handlelength=1.3,columnspacing=.8)
    regret=np.full((len(ORDER),len(SCENES)),np.nan);oracle=np.full_like(regret,np.nan)
    for j,scene in enumerate(SCENES):
        costs={arm:by[(scene,arm)]['agents'][1]['cost'] for arm in ORDER if (scene,arm) in by}
        pred={arm:manifests[scene]['predictions'][arm][1]['cost'] for arm in ORDER}
        for i,arm in enumerate(ORDER):
            if arm in costs:regret[i,j]=costs[arm]-min(costs.values())
            oracle[i,j]=pred[arm]-min(pred.values())
    image=b.imshow(regret,cmap='Purples',vmin=0,vmax=np.nanmax(regret),aspect='auto')
    for i in range(len(ORDER)):
        for j in range(len(SCENES)):
            v=regret[i,j];b.text(j,i,'—' if np.isnan(v) else f'{v:.0f}',ha='center',va='center',fontsize=fs(6.3),color='white' if v>.55*np.nanmax(regret) else 'black')
            if oracle[i,j]>.5:b.add_patch(plt.Rectangle((j-.5,i-.5),1,1,fill=False,ec='#C62828',lw=.8))
    b.set_xticks(range(len(SCENES)),[SCENE_LABEL[s] for s in SCENES],fontsize=fs(6.3));b.set_yticks(range(len(ORDER)),[LABEL[x] for x in ORDER],fontsize=fs(6.3))
    b.tick_params(length=0);[s.set_visible(False) for s in b.spines.values()];b.yaxis.tick_left()
    b.set_title('Carrier excess cost\nvs best tested arm',fontsize=fs(6.6),pad=2,linespacing=1.1)
    kb=[np.mean([by[(s,arm)]['agents'][0]['bytes']/1e3 for s in SCENES if (s,arm) in by]) for arm in ORDER]
    c=fig.add_axes([.885,.2,.1,.66],sharey=b);c.barh(range(len(ORDER)),kb,color='#B0B5BB',height=.6);c.tick_params(labelleft=False,length=2);c.set_title('scout kB',fontsize=fs(6.3),pad=2)
    c.spines['top'].set_visible(False);c.spines['right'].set_visible(False);c.locator_params(axis='x',nbins=2);c.tick_params(axis='x',labelsize=fs(5.8))
    b.text(0,-.2,'red outline: oracle predicts excess cost > 0.5',transform=b.transAxes,fontsize=fs(5.8),color='#C62828',va='top',ha='left')
    a.set_title(f'Carrier routes, scene "{SCENE_LABEL[args.scene]}"',fontsize=fs(6.8),pad=2)
    for ax,label,x in ((a,'a',0),(b,'b',-.98)):ax.text(x,1.02,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom')
    save_figure(fig,'fig7_terrain',sources,notes=dict(scene=args.scene,definition='regret = carrier cost minus the best tested arm in the same scene (not a globally optimal policy)',regret=regret.tolist(),oracle_regret=oracle.tolist(),order=ORDER,scenes=SCENES,kB=kb))
    plt.close(fig)


if __name__=='__main__':main()
