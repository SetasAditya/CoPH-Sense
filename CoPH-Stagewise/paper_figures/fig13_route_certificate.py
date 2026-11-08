"""F13: from identical evidence, field packets choose better routes and know when they might be wrong."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from .style import setup_style,save_figure,fs,ROOT

DEC=ROOT/'results/reconstruction_decision_v1';CERT=ROOT/'results/route_certificate_v1'
REPS=('field','costmap','raw');LABEL=dict(field='CoPH field',costmap='graded costmap',raw='raw readings')
COLOR=dict(field='#2a78d6',costmap='#eb6834',raw='#8E949C');INK,MUTED,GRID='#1f2328','#6B7178','#E8EAED'


def main(argv=None):
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    dec=json.loads((DEC/'summary.json').read_text());cert=json.loads((CERT/'summary.json').read_text())
    setup_style(5.5);fig=plt.figure(figsize=(5.5,2.15))
    gs=fig.add_gridspec(1,3,width_ratios=[1.15,.9,1.15],left=.1,right=.975,top=.83,bottom=.2,wspace=.3)
    a,b,c=(fig.add_subplot(gs[k]) for k in range(3))
    strata=(('near_anchors','near'),('between','between'),('far','far'))
    w=.26
    for k,rep in enumerate(REPS):
        v=[dec[s][rep]['mean_regret'] for s,_ in strata]
        a.bar(np.arange(3)+(k-1)*w,v,width=w-.03,color=COLOR[rep],label=LABEL[rep])
    a.set_xticks(range(3));a.set_xticklabels([l for _,l in strata]);a.set_xlabel('route distance from evidence',labelpad=1)
    a.set_ylabel('mean regret',labelpad=1);a.grid(axis='y',color=GRID,lw=.5);a.set_axisbelow(True)
    a.set_title('better choices (12/12 groups)',fontsize=fs(6.6),pad=3);a.legend(loc='upper left',fontsize=fs(5.6),handlelength=1,borderaxespad=.1)
    y=np.arange(3)
    for k,rep in enumerate(REPS):b.barh(k,100*cert[rep]['calibration90'],height=.55,color=COLOR[rep]);b.text(100*cert[rep]['calibration90']-2,k,f"{100*cert[rep]['calibration90']:.0f}%",ha='right',va='center',color='white',fontsize=fs(5.8),weight='bold')
    b.axvline(90,color=INK,lw=.8,ls=(0,(3,2)));b.text(90,-.62,'nominal 90%',ha='center',va='bottom',fontsize=fs(5.6),color=INK)
    b.set(xlim=(50,100),yticks=[],ylim=(2.5,-.8));b.spines['left'].set_visible(False)
    b.set_xlabel('true cost gap inside\nstated 90% interval',labelpad=1);b.set_title('honest uncertainty',fontsize=fs(6.6),pad=3)
    for k,rep in enumerate(REPS):
        r=cert[rep];c.plot([100*r['wrong_when_certified'],100*r['wrong_when_uncertain']],[k,k],color='#D5D9DE',lw=1.4,zorder=1)
        c.plot(100*r['wrong_when_certified'],k,'o',ms=fs(4.2),color=COLOR[rep],zorder=3)
        c.plot(100*r['wrong_when_uncertain'],k,'o',ms=fs(4.2),mfc='white',mec=COLOR[rep],mew=1.1,zorder=3)
        c.text(100*r['wrong_when_certified'],k-.3,f"{100*r['wrong_when_certified']:.1f}%",ha='center',va='bottom',fontsize=fs(5.4),color=INK)
        c.text(100*r['wrong_when_uncertain'],k+.32,f"{100*r['wrong_when_uncertain']:.0f}%, {100*r['certified_share']:.0f}% certified",ha='center',va='top',fontsize=fs(5.2),color=MUTED)
    c.axvline(5,color=INK,lw=.6,ls=(0,(3,2)));c.set(xlim=(-1.5,40),ylim=(2.5,-.8),yticks=[]);c.spines['left'].set_visible(False)
    c.set_xlabel('wrong route choices (%)',labelpad=1);c.grid(axis='x',color=GRID,lw=.5);c.set_axisbelow(True)
    c.set_title('● certified   ○ flagged uncertain',fontsize=fs(6.2),pad=3)
    for ax,label,x in ((a,'a',-.32),(b,'b',-.08),(c,'c',-.06)):ax.text(x,1.04,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom')
    save_figure(fig,'fig13_route_certificate',[DEC/'summary.json',CERT/'summary.json'],notes=dict(cert=cert))
    plt.close(fig)


if __name__=='__main__':main()
