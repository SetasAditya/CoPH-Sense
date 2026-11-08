"""F14: symmetric encounters. Shared protection alone gridlocks a 4-agent swap; deterministic priority clears it."""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from .style import setup_style,save_figure,fs,ROOT

DEFAULT=ROOT/'results/native_liveness_range_v1'
AGENTS=('#2a78d6','#eb6834','#1baf7a','#6250d6');INK,MUTED,GRID='#1f2328','#6B7178','#E8EAED'
FAMILIES=(('pair_exact','head-on'),('pair_swap','offset head-on'),('circle_swap','4-agent swap'),('crossing','crossing'))


def main():
    rows=json.loads((DEFAULT/'rows.json').read_text());s=json.loads((DEFAULT/'summary.json').read_text())['summary']
    setup_style(5.5);fig=plt.figure(figsize=(5.5,2.5))
    gs=fig.add_gridspec(1,3,width_ratios=[1,1,1.55],left=.02,right=.975,top=.8,bottom=.17,wspace=.42)
    seed=200
    for k,(arm,title) in enumerate((('coph_online','shared protection only'),('coph_online_priority_range','+ priority, braking-range layer'))):
        ax=fig.add_subplot(gs[k]);r=next(x for x in rows if x['family']=='circle_swap' and x['arm']==arm and x['seed']==seed);q=np.array(r['path'])
        for i in range(4):
            ax.plot(q[:,i,0],q[:,i,1],color=AGENTS[i],lw=1.1)
            ax.add_patch(Circle(q[-1,i],.15,fc=AGENTS[i],ec='white',lw=.6,alpha=.9,zorder=3))
            ax.plot(*r['starts'][i],'o',ms=fs(2.6),mfc='white',mec=AGENTS[i],mew=.9,zorder=4)
            ax.plot(*r['goals'][i],'x',ms=fs(3.4),color=AGENTS[i],mew=1.,zorder=4)
        ax.set(xlim=(-1.3,1.3),ylim=(-1.3,1.3),aspect='equal',xticks=[],yticks=[])
        for sp in ax.spines.values():sp.set_visible(False)
        ax.set_facecolor('#F7F8FA');ax.set_title(title,fontsize=fs(6.6),pad=3)
        ax.text(.5,-.04,'gridlocked at the centre' if not r['success'] else f"all goals by step {r['time_to_goals']}",transform=ax.transAxes,ha='center',va='top',fontsize=fs(5.8),color=MUTED)
        ax.text(0,1.13,'ab'[k],transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom')
    c=fig.add_subplot(gs[2]);y=np.arange(len(FAMILIES));h=.2
    for j,(arm,color,lab) in enumerate((('attraction','#B8BDC4','attraction'),('attraction_priority_range','#8E949C','attraction + fix'),
                                         ('coph_online','#9EC3EE','CoPH online'),('coph_online_priority_range','#2a78d6','CoPH online + fix'))):
        v=[s[f'{f}/{arm}']['success'] for f,_ in FAMILIES]
        c.barh(y+(j-1.5)*h,v,height=h*.92,color=color,label=lab)
        for yy,vv,f in zip(y,v,FAMILIES):
            n=s[f'{f[0]}/{arm}']['new_contacts']
            if vv==0:c.text(.5,yy+(j-1.5)*h,'0',va='center',fontsize=fs(5),color=INK)
            if n:c.text(vv+.6,yy+(j-1.5)*h,f'{n} contact'+('s' if n>1 else ''),va='center',fontsize=fs(5),color='#C62828')
    c.set_yticks(y);c.set_yticklabels([t for _,t in FAMILIES]);c.invert_yaxis();c.tick_params(axis='y',length=0);c.spines['left'].set_visible(False)
    c.set(xlim=(0,40),xticks=[0,10,20,30]);c.grid(axis='x',color=GRID,lw=.5);c.set_axisbelow(True);c.set_xlabel('episodes reaching all goals /30',labelpad=1)
    c.legend(loc='lower center',bbox_to_anchor=(.42,1.0),ncol=2,fontsize=fs(5.4),handlelength=1,columnspacing=.8,labelspacing=.2,borderaxespad=.2)
    c.text(-.36,1.13,'c',transform=c.transAxes,fontsize=fs(8.5),weight='bold',va='bottom',ha='right')
    save_figure(fig,'fig14_liveness',[DEFAULT/'summary.json',DEFAULT/'rows.json'])
    plt.close(fig)


if __name__=='__main__':main()
