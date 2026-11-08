"""F9: where the transferred CoPH controller beats the published phMARL baseline (native VMAS)."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .style import setup_style,save_figure,fs
from .native_data import DEFAULT,ARMS,LABEL,COLOR,load,drag_traces,PRIOR,NATIVE_DRAG

ONLINE='coph_online_protected'
RIVALS=(('phmarl','o','phMARL'),('attraction_protected','s','protected attraction'),('coph_frozen_protected','^','frozen CoPH'))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=DEFAULT);args=ap.parse_args(argv)
    data,sources=load(args.results);traces,more=drag_traces(args.results);sources+=more
    setup_style(5.5);fig=plt.figure(figsize=(5.5,2.35))
    gs=fig.add_gridspec(1,3,width_ratios=[1,1.25,.85],left=.085,right=.99,top=.86,bottom=.3,wspace=.55)
    a,b,c=(fig.add_subplot(gs[k]) for k in range(3))
    online=np.array([r['metrics']['mean_agent_return'] for r in data[ONLINE]]);wins={}
    for arm,marker,short in RIVALS:
        rival=np.array([r['metrics']['mean_agent_return'] for r in data[arm]]);wins[arm]=int(np.sum(online>rival))
        a.scatter(rival,online,s=fs(12),marker=marker,facecolor=COLOR[arm],edgecolor='black',lw=.3,zorder=3,label=f'vs {short} ({wins[arm]}/20)')
    lo=min(min(r['metrics']['mean_agent_return'] for arm,*_ in RIVALS for r in data[arm]),online.min())*1.05;hi=max(online.max(),-40)
    a.plot([lo,hi],[lo,hi],color='#9AA1A9',lw=.7,ls=(0,(3,2)),zorder=1);a.set(xlim=(lo,hi),ylim=(lo,hi))
    a.set_xlabel('baseline return',labelpad=1);a.set_ylabel('online CoPH return',labelpad=1)
    a.text(.97,.04,f"online wins {sum(wins.values())}/{20*len(RIVALS)}\n\u25cf phMARL  \u25a0 prot. attr.\n\u25b2 frozen CoPH",transform=a.transAxes,ha='right',va='bottom',fontsize=fs(5.8),linespacing=1.2)
    a.set_title('Per-episode return',fontsize=fs(6.8),pad=2)
    steps=np.arange(1,401)
    for arm in ARMS:
        curves=np.array([np.cumsum(r['rewards'].mean(1)) for r in data[arm]]);mid=curves.mean(0);q1,q3=np.percentile(curves,[25,75],axis=0)
        bold=arm==ONLINE;b.plot(steps,mid,color=COLOR[arm],lw=1.6 if bold else 1.,zorder=4 if bold else 3,label=LABEL[arm])
        b.fill_between(steps,q1,q3,color=COLOR[arm],alpha=.14 if bold else .07,lw=0,zorder=2)
    b.set(xlim=(0,400),xlabel='step');b.set_ylabel('cumulative reward',labelpad=1);b.grid(color='#E8EAED',lw=.5)
    b.set_title('Return accumulated over the episode',fontsize=fs(6.8),pad=2)
    window=221;med=np.median(traces[:,:window],0);p10,p90=np.percentile(traces[:,:window],[10,90],axis=0)
    c.fill_between(range(window),p10,p90,step='post',color=COLOR[ONLINE],alpha=.18,lw=0);c.step(range(window),med,where='post',color=COLOR[ONLINE],lw=1.4)
    c.axhline(NATIVE_DRAG,color='black',lw=.7,ls=(0,(3,2)));c.axhline(PRIOR,color='#9AA1A9',lw=.7,ls=(0,(1,1.5)))
    c.text(215,NATIVE_DRAG-.012,'native drag 0.25',ha='right',va='top',fontsize=fs(5.8));c.text(215,PRIOR+.01,'prior 0.058',ha='right',va='bottom',fontsize=fs(5.8),color='#6B7178')
    c.set(xlim=(0,220),ylim=(0,.3),xlabel='step');c.set_ylabel('identified drag',labelpad=1);c.set_title('Online identification',fontsize=fs(6.8),pad=2);c.grid(color='#E8EAED',lw=.5)
    for ax,label,x in ((a,'a',-.34),(b,'b',-.26),(c,'c',-.42)):ax.text(x,1.03,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom')
    handles=[Line2D([],[],color=COLOR[arm],lw=1.6 if arm==ONLINE else 1.,label=LABEL[arm]) for arm in ARMS]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=fs(6),handlelength=1.5,columnspacing=1.2,bbox_to_anchor=(.55,-.01))
    reached=int(np.sum(np.all(np.isclose(traces[:,16:],NATIVE_DRAG,atol=1e-6),axis=1)))
    first=[int(np.argmax(np.isclose(t,NATIVE_DRAG,atol=1e-6))) for t in traces];all_by=max(first)
    save_figure(fig,'fig9_native',sources,notes=dict(wins=wins,online_mean=float(online.mean()),agent_episodes=len(traces),reached_native_by_step16=reached,all_reached_by_step=all_by,median_step=float(np.median(first))))
    plt.close(fig)


if __name__=='__main__':main()
