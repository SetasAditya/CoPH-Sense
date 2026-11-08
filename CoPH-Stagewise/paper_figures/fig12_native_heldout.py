"""F12: held-out native starts (seeds 85000-85049) with allocation, targets and protection matched across arms."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .style import setup_style,save_figure,fs,ROOT

DEFAULT=ROOT/'results/native_heldout_v1'
ARMS=('coph_online_static_protected','coph_known_static_protected','coph_frozen_static_protected',
      'phmarl_assigned_static_protected','phmarl_assigned','phmarl')
LABEL=dict(coph_online_static_protected='CoPH, online drag',coph_known_static_protected='CoPH, known drag 0.25',
           coph_frozen_static_protected='CoPH, prior drag 0.058',phmarl_assigned_static_protected='phMARL + matched layers',
           phmarl_assigned='phMARL + allocation',phmarl='phMARL (published)')
COLOR=dict(coph_online_static_protected='#2a78d6',coph_known_static_protected='#1baf7a',coph_frozen_static_protected='#6250d6',
           phmarl_assigned_static_protected='#eb6834',phmarl_assigned='#B8BDC4',phmarl='#B8BDC4')
INK,MUTED,GRID='#1f2328','#6B7178','#E8EAED'


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=DEFAULT);args=ap.parse_args(argv)
    s=json.loads((args.results/'summary.json').read_text());seeds=list(range(85000,85050))
    ret={a:np.array([json.loads((args.results/f'{a}_seed{k}_metrics.json').read_text())['mean_agent_return'] for k in seeds]) for a in ARMS}
    setup_style(5.5);fig=plt.figure(figsize=(5.5,3.7))
    gs=fig.add_gridspec(2,2,width_ratios=[2.6,1],height_ratios=[1.25,1],left=.25,right=.975,top=.93,bottom=.08,wspace=.08,hspace=.5)
    a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[0,1]);c=fig.add_subplot(gs[1,:]);rng=np.random.RandomState(0)
    for y,arm in enumerate(ARMS):
        v=ret[arm];jit=rng.uniform(-.17,.17,len(v))
        a.scatter(v,y+jit,s=fs(5),color=COLOR[arm],alpha=.55,lw=0,zorder=3)
        q1,med,q3=np.percentile(v,[25,50,75]);a.plot([q1,q3],[y,y],color=INK,lw=2.2,solid_capstyle='butt',zorder=4)
        a.plot(med,y,'o',ms=fs(3.2),mfc='white',mec=INK,mew=.9,zorder=5)
    a.set_yticks(range(len(ARMS)));a.set_yticklabels([LABEL[x] for x in ARMS]);a.invert_yaxis();a.tick_params(axis='y',length=0)
    a.spines['left'].set_visible(False);a.grid(axis='x',color=GRID,lw=.5);a.set_xlabel('episode return (higher is better)',labelpad=1)
    a.set_title('return on 50 unused starts',fontsize=fs(6.6),pad=3)
    for y,arm in enumerate(ARMS):
        n=s['summary'][arm]['full_coverage'];b.barh(y,n,height=.56,color=COLOR[arm],edgecolor='white',lw=.6)
        b.text(n+1.2,y,f'{n}',va='center',fontsize=fs(5.8),color=INK)
    b.set(xlim=(0,62),yticks=[]);b.invert_yaxis();b.spines['left'].set_visible(False);b.set_xticks([0,25,50]);b.set_xlabel('full coverage /50',labelpad=1)
    b.set_title('task solved',fontsize=fs(6.6),pad=3)
    p=s['paired'][2];on=ret[ARMS[0]];base=ret['phmarl_assigned_static_protected'];order=np.argsort(base)
    for i,k in enumerate(order):c.plot([i,i],[base[k],on[k]],color='#D5D9DE',lw=.7,zorder=1)
    c.scatter(range(50),base[order],s=fs(5),color=COLOR['phmarl_assigned_static_protected'],lw=0,zorder=3,label='phMARL + layers')
    c.scatter(range(50),on[order],s=fs(5),color=COLOR[ARMS[0]],lw=0,zorder=3,label='CoPH online')
    c.set_xticks([]);c.set_xlabel('starts, sorted by baseline',labelpad=2);c.grid(axis='y',color=GRID,lw=.5)
    c.set_title(f"same start: +{p['difference']:.0f} [{p['interval95'][0]:.0f}, {p['interval95'][1]:.0f}], {p['wins']}/50 wins",fontsize=fs(6.6),pad=3)
    c.set_ylabel('return',labelpad=1)
    c.legend(loc='lower right',ncol=2,fontsize=fs(5.4),handletextpad=.1,borderaxespad=.1,markerscale=1.4)
    for ax,label,x in ((a,'a',-.02),(b,'b',-.03),(c,'c',-.02)):ax.text(x,1.04,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom',ha='right')
    save_figure(fig,'fig12_native_heldout',[args.results/'summary.json'],notes=dict(paired=s['paired'],gate=s['gate']))
    plt.close(fig)


if __name__=='__main__':main()
