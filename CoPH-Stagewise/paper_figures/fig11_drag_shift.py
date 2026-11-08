"""F11: a payload change mid-mission. Online identification re-tracks drag; the fixed correct model goes stale."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .style import setup_style,save_figure,fs,ROOT

DEFAULT=ROOT/'results/native_drag_shift_v1'
ARMS=('coph_online_static_protected','coph_known_static_protected','coph_frozen_static_protected','phmarl_assigned_static_protected')
LABEL=dict(coph_online_static_protected='CoPH, online drag',coph_known_static_protected='CoPH, drag fixed at 0.25',
           coph_frozen_static_protected='CoPH, prior drag 0.058',phmarl_assigned_static_protected='phMARL (matched)')
COLOR=dict(coph_online_static_protected='#2a78d6',coph_known_static_protected='#1baf7a',coph_frozen_static_protected='#6250d6',
           phmarl_assigned_static_protected='#8E949C')
INK,MUTED,GRID='#1f2328','#6B7178','#E8EAED';SWITCH=200;PRIOR=-np.expm1(-.06)


def load(root):
    rows=json.loads((root/'rows.json').read_text());curves={};drag={}
    for r in rows:
        tag='%s_seed%d_d%03d'%(r['protocol'],r['seed'],round(r['d2']*100))
        with np.load(root/('%s_%s.npz'%(r['protocol'],tag))) as d:curves.setdefault((r['protocol'],r['d2']),[]).append(d['rewards'].reshape(-1,4).mean(1))
        if r['drag_updates'] is not None and r['protocol']=='coph_online_static_protected':
            for i in range(4):
                t=np.full(400,PRIOR)
                for u in r['drag_updates']:
                    if u['agent']==i:t[u['step']:]=u['committed']
                drag.setdefault(r['d2'],[]).append(t)
    return rows,curves,drag


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=DEFAULT);args=ap.parse_args(argv)
    rows,curves,drag=load(args.results);summary=json.loads((args.results/'summary.json').read_text())
    setup_style(5.5);fig=plt.figure(figsize=(5.5,3.7))
    gs=fig.add_gridspec(2,3,height_ratios=[1,.62],left=.09,right=.985,top=.93,bottom=.1,wspace=.42,hspace=.62)
    axes=[fig.add_subplot(gs[0,k]) for k in range(3)]+[fig.add_subplot(gs[1,1:])];steps=np.arange(400)
    def smooth(y,w=9):return np.convolve(np.pad(y,(w//2,w//2),mode='edge'),np.ones(w)/w,mode='valid')
    for ax,d2,title in ((axes[0],.45,'heavier payload: drag 0.25 → 0.45'),(axes[1],.10,'lighter payload: drag 0.25 → 0.10')):
        ax.axvspan(SWITCH,400,color='#F4F5F7',lw=0,zorder=0);ax.axvline(SWITCH,color=MUTED,lw=.6,ls=(0,(2,2)))
        for arm in ARMS[::-1]:
            c=np.array(curves[arm,d2]);bold=arm==ARMS[0]
            ax.plot(steps,smooth(c.mean(0)),color=COLOR[arm],lw=1.7 if bold else 1.,zorder=4 if bold else 3)
        ax.set(xlim=(0,400),xticks=[0,200,400],ylim=(-1.6,0.2),yticks=[-1.5,-1,-.5,0]);ax.set_xlabel('step',labelpad=1);ax.grid(axis='y',color=GRID,lw=.5)
        ax.set_title(title,fontsize=fs(6.6),pad=3,color=INK)
        ax.text(SWITCH+6,.1,'payload change',fontsize=fs(5.5),color=MUTED,va='center')
    axes[0].set_ylabel('team reward per step',labelpad=1)
    c=axes[2]
    for d2,ls in ((.45,'-'),(.10,'-')):
        t=np.array(drag[d2]);med=np.median(t,0);lo,hi=np.percentile(t,[10,90],axis=0)
        c.fill_between(steps,lo,hi,step='post',color=COLOR[ARMS[0]],alpha=.16,lw=0);c.step(steps,med,where='post',color=COLOR[ARMS[0]],lw=1.4)
        c.plot([SWITCH,400],[d2,d2],color=INK,lw=.7,ls=(0,(3,2)))
        c.text(395,d2+(.015 if d2>.3 else -.015),f'true {d2:.2f}',ha='right',va='bottom' if d2>.3 else 'top',fontsize=fs(5.6),color=INK)
    c.plot([0,SWITCH],[.25,.25],color=INK,lw=.7,ls=(0,(3,2)));c.text(5,.262,'true 0.25',fontsize=fs(5.6),color=INK,va='bottom')
    c.plot([0,SWITCH],[PRIOR,PRIOR],color=MUTED,lw=.6,ls=(0,(1,1.5)));c.text(5,PRIOR+.008,'prior 0.058',va='bottom',fontsize=fs(5.6),color=MUTED)
    c.axvline(SWITCH,color=MUTED,lw=.6,ls=(0,(2,2)));c.axvspan(SWITCH,400,color='#F4F5F7',lw=0,zorder=0)
    c.set(xlim=(0,400),xticks=[0,200,400],ylim=(0,.52));c.set_xlabel('step',labelpad=1);c.set_ylabel('identified drag',labelpad=1)
    c.set_title('online drag estimate',fontsize=fs(6.6),pad=3);c.grid(axis='y',color=GRID,lw=.5)
    d=axes[3];labels=[];y=0
    for rival,short in ((ARMS[1],'fixed 0.25'),(ARMS[2],'prior 0.058'),(ARMS[3],'phMARL')):
        for d2,marker in ((.45,'o'),(.10,'s')):
            p=summary['conditions'][str(d2)]['paired'][f'online_minus_{rival}_phase2_return']
            d.plot(p['interval95'],[y,y],color=COLOR[ARMS[0]],lw=1.6,solid_capstyle='round')
            d.plot(p['mean'],y,marker=marker,ms=fs(3.6),color='white',mec=COLOR[ARMS[0]],mew=1.1,zorder=3)
            d.text(p['interval95'][1]+2,y,f"{p['mean']:.0f}  ({p['wins']}/30 wins)",va='center',fontsize=fs(5.4),color=MUTED)
            y-=.42
        labels.append((y+.63,'vs '+short));y-=.5
    d.axvline(0,color=INK,lw=.6);d.set_yticks([l[0] for l in labels]);d.set_yticklabels([l[1] for l in labels])
    d.tick_params(axis='y',length=0);d.spines['left'].set_visible(False);d.set_xlim(0,105);d.grid(axis='x',color=GRID,lw=.5)
    d.set_xlabel('phase-2 return gain of online CoPH (paired, 95% CI)',labelpad=1)
    d.legend(handles=[Line2D([],[],marker='o',ls='',mfc='white',mec=COLOR[ARMS[0]],label='→ 0.45'),Line2D([],[],marker='s',ls='',mfc='white',mec=COLOR[ARMS[0]],label='→ 0.10')],
             loc='upper right',fontsize=fs(5.6),handletextpad=.2,borderaxespad=.1,title='drag after change',title_fontsize=fs(5.6))
    for ax,label in zip(axes,'abcd'):ax.text(-.24 if label!='d' else -.3,1.04,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom')
    handles=[Line2D([],[],color=COLOR[a],lw=1.7 if a==ARMS[0] else 1.,label=LABEL[a]) for a in ARMS]
    fig.legend(handles=handles,loc='center left',ncol=1,fontsize=fs(6),handlelength=1.6,bbox_to_anchor=(.0,.2),labelspacing=.7)
    save_figure(fig,'fig11_drag_shift',[args.results/'summary.json',args.results/'rows.json'],notes=dict(conditions={k:v['gate'] for k,v in summary['conditions'].items()}))
    plt.close(fig)


if __name__=='__main__':main()
