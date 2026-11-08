"""F4: reciprocal warnings save more than the two one-way channels combined."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .style import setup_style,save_figure,fs,AGENT,ARM_COLOR,MESSAGE,ROOT
from .data import results,wasted

DEFAULT=[ROOT/'results'/'crossing_traverses_v1']
CHANNELS=('one_way_01','one_way_10','reciprocal_fixed')


def savings(dirs):
    """Paired savings per (results dir, seed): measured and oracle, per agent."""
    rows=[];sources=[]
    for d in dirs:
        episodes,src=results(d);sources+=src
        # Predictions frozen in the manifest before the runs, never recomputed afterwards.
        frozen=json.loads((Path(d)/'manifest.json').read_text())['predictions']
        for seed in sorted({s for s,_ in episodes}):
            if not all((seed,a) in episodes for a in ('disabled',)+CHANNELS):continue
            base=wasted(episodes[(seed,'disabled')]);pbase=frozen['disabled']
            row=dict(scene=Path(d).name,seed=seed)
            for arm in CHANNELS:
                w=wasted(episodes[(seed,arm)]);p=frozen[arm]
                row[arm]=[base[i]-w[i] for i in (0,1)];row['oracle_'+arm]=[pbase[str(i)]['wasted']-p[str(i)]['wasted'] for i in (0,1)]
            rows.append(row)
    return rows,sources


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,nargs='+',default=DEFAULT);args=ap.parse_args(argv)
    rows,sources=savings(args.results);setup_style(5.5)
    fig,(a,b)=plt.subplots(1,2,figsize=(5.5,1.85),gridspec_kw=dict(width_ratios=[1.15,1],left=.085,right=.99,top=.86,bottom=.2,wspace=.32))
    team=lambda r,k:sum(r[k]);labels=['A→B\nonly','B→A\nonly','sum of\none-way','reciprocal']
    measured=[[team(r,'one_way_01') for r in rows],[team(r,'one_way_10') for r in rows],[team(r,'one_way_01')+team(r,'one_way_10') for r in rows],[team(r,'reciprocal_fixed') for r in rows]]
    predicted=[[team(r,'oracle_one_way_01') for r in rows],[team(r,'oracle_one_way_10') for r in rows],[team(r,'oracle_one_way_01')+team(r,'oracle_one_way_10') for r in rows],[team(r,'oracle_reciprocal_fixed') for r in rows]]
    colors=[ARM_COLOR['one_way_01'],ARM_COLOR['one_way_10'],'#CFC4E6',MESSAGE]
    for k,(m,p) in enumerate(zip(measured,predicted)):
        a.bar(k,np.mean(m),.66,color=colors[k],zorder=2,hatch='////' if k==2 else None,edgecolor='white' if k==2 else None,lw=0)
        a.scatter(np.full(len(m),k)+np.linspace(-.12,.12,len(m)),m,s=fs(4),color='black',zorder=3,lw=0)
        a.plot([k-.36,k+.36],[np.mean(p)]*2,color='black',lw=1.,ls=(0,(2,1)),zorder=4)
    ratio=np.mean(measured[3])/max(np.mean(measured[2]),1e-9)
    a.annotate('',xy=(3,np.mean(measured[3])),xytext=(2,np.mean(measured[2])),arrowprops=dict(arrowstyle='-|>',color='#333',lw=.8,connectionstyle='arc3,rad=-.35',mutation_scale=fs(6)))
    a.text(2.5,np.mean(measured[3])*1.08,f'×{ratio:.1f}',ha='center',va='bottom',fontsize=fs(7),weight='bold')
    a.set_xticks(range(4),labels,fontsize=fs(6.3));a.set_ylabel('team detour saved (m)',labelpad=1);a.set_ylim(0,max(map(np.mean,measured+predicted))*1.3)
    a.grid(axis='y',color='#E3E5E8',lw=.5,zorder=0);a.set_title('Team saving vs. no communication',fontsize=fs(6.8),pad=2)
    width=.36
    for i in (0,1):
        for k,arm in enumerate(CHANNELS):
            m=[r[arm][i] for r in rows];p=[r['oracle_'+arm][i] for r in rows];x=k+(i-.5)*width
            b.bar(x,np.mean(m),width*.92,color=AGENT[i],zorder=2)
            b.scatter(np.full(len(m),x),m,s=fs(4),color='black',zorder=3,lw=0)
            b.plot([x-width*.46,x+width*.46],[np.mean(p)]*2,color='black',lw=1.,ls=(0,(2,1)),zorder=4)
    b.set_xticks(range(3),['A→B only','B→A only','reciprocal'],fontsize=fs(6.3));b.set_ylabel('detour saved (m)',labelpad=1)
    b.set_ylim(0,a.get_ylim()[1]*.6);b.grid(axis='y',color='#E3E5E8',lw=.5,zorder=0);b.set_title('Who benefits',fontsize=fs(6.8),pad=2)
    b.legend(handles=[Line2D([],[],color=AGENT[0],lw=4,label='A'),Line2D([],[],color=AGENT[1],lw=4,label='B'),
                      Line2D([],[],color='black',lw=1.,ls=(0,(2,1)),label='oracle'),Line2D([],[],marker='o',color='black',lw=0,ms=2,label='per seed')],
             loc='upper left',fontsize=fs(6),ncol=2,handlelength=1.3,columnspacing=.8,borderaxespad=0)
    for ax,label,x in ((a,'a',-.17),(b,'b',-.2)):ax.text(x,1.02,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom')
    save_figure(fig,'fig4_superadditivity',sources,notes=dict(rows=rows,ratio=ratio,saving='paired difference in detour (path minus true shortest) against the no-communication run of the same seed'))
    plt.close(fig)


if __name__=='__main__':main()
