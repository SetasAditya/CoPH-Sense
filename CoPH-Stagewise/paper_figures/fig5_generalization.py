"""F5: measured vs oracle-predicted saving across the complex-maze family."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .style import setup_style,save_figure,fs,MESSAGE,ARM_COLOR
from .fig4_superadditivity import savings
from .pending import placeholder


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,nargs='*',default=[]);args=ap.parse_args(argv)
    rows,sources=savings(args.results) if args.results else ([],[])
    scenes={r['scene'] for r in rows}
    if len(scenes)<3:
        return placeholder('fig5_generalization','Generalization across generated mazes','complex-maze family runs (plan gate 2/4); needs ≥3 mazes')
    setup_style(5.5);fig,(a,b)=plt.subplots(1,2,figsize=(5.5,2.15),gridspec_kw=dict(width_ratios=[1,1],left=.09,right=.99,top=.86,bottom=.2,wspace=.32))
    for arm,marker in (('one_way_01','v'),('one_way_10','^'),('reciprocal_fixed','o')):
        x=[sum(r['oracle_'+arm]) for r in rows];y=[sum(r[arm]) for r in rows]
        a.scatter(x,y,s=fs(10),marker=marker,color=ARM_COLOR[arm],edgecolor='black',lw=.3,zorder=3,label=arm.replace('_fixed','').replace('one_way_01','A→B').replace('one_way_10','B→A'))
    top=max(max(sum(r['oracle_reciprocal_fixed']),sum(r['reciprocal_fixed'])) for r in rows)*1.1
    a.plot([0,top],[0,top],color='#9AA1A9',lw=.7,ls=(0,(3,2)),zorder=1);a.set(xlim=(0,top),ylim=(0,top),xlabel='oracle-predicted saving (m)',ylabel='measured saving (m)')
    a.legend(fontsize=fs(6),loc='upper left',handlelength=1);a.set_title('Measured vs. frozen oracle prediction',fontsize=fs(6.8),pad=2)
    # One point per scene: seeds of a scene are averaged so no maze counts twice.
    byscene={s:[r for r in rows if r['scene']==s] for s in sorted(scenes)}
    rec=np.array([np.mean([sum(r['reciprocal_fixed']) for r in rs]) for rs in byscene.values()])
    one=np.array([np.mean([sum(r['one_way_01'])+sum(r['one_way_10']) for r in rs]) for rs in byscene.values()])
    b.scatter(one,rec,s=fs(10),color=MESSAGE,edgecolor='black',lw=.3,zorder=3);lim=max(rec.max(),one.max())*1.1
    b.plot([0,lim],[0,lim],color='#9AA1A9',lw=.7,ls=(0,(3,2)));b.set(xlim=(0,lim),ylim=(0,lim),xlabel='sum of one-way savings (m)',ylabel='reciprocal saving (m)')
    b.text(.97,.05,f'{int(np.sum(rec>one+.5))} of {len(rec)} scenes superadditive',transform=b.transAxes,ha='right',fontsize=fs(6.3));b.set_title('Reciprocal vs. sum of one-way savings',fontsize=fs(6.8),pad=2)
    for ax,label in ((a,'a'),(b,'b')):ax.text(-.16,1.02,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom')
    save_figure(fig,'fig5_generalization',sources,notes=dict(rows=rows));plt.close(fig)


if __name__=='__main__':main()
