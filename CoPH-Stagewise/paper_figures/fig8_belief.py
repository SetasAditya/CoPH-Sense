"""F8: what the carrier believes about a corridor it has never entered, by sharing mode."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from .style import setup_style,save_figure,fs,MESSAGE
from .pending import placeholder
from .hazard_data import ROW,true_row

SHOW=dict(none=('#9AA1A9','no communication'),field=(MESSAGE,'learned env. field'),raw=('#B39DDB','raw readings'),occupancy_0p25=('#E69F00','occupancy τ=0.25'),decision=('#009E73',"sender's decision"))
YMAX=.6


def limit(spec):
    """Exposure above background at which the carrier should detour (route arithmetic)."""
    w=spec['risk_weights'][1];return 7*(1+4*w*.02)/(4*w)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,nargs='*',default=[]);ap.add_argument('--scenes',nargs='+',default=['double_mild','moderate']);args=ap.parse_args(argv)
    if not args.results:return placeholder('fig8_belief','Carrier belief about an unvisited corridor','graded-hazard pilot runs')
    from .hazard_data import results
    episodes,manifests,sources=results(args.results[0]);by={(e['scene'],e['arm']):e for e in episodes}
    setup_style(5.5);fig,axes=plt.subplots(1,len(args.scenes),figsize=(5.5,2.1),sharey=True,gridspec_kw=dict(left=.1,right=.99,top=.87,bottom=.36,wspace=.08))
    x=ROW[:,0];notes={}
    for k,(ax,scene) in enumerate(zip(axes,args.scenes)):
        spec=by[(scene,'field')]['spec'];truth=true_row(spec);ax.fill_between(x,truth,color='#C62828',alpha=.15,lw=0);ax.plot(x,truth,color='#C62828',lw=1.,label='true risk')
        exposures={}
        for arm,(color,label) in SHOW.items():
            if (scene,arm) not in by:continue
            b=np.array(by[(scene,arm)]['belief']['values']);exposures[arm]=float(np.trapezoid(np.clip(b-.02,0,None),x))
            ax.plot(x,np.minimum(b,YMAX*.97),color=color,lw=1.3 if arm=='field' else 1.,ls='-' if arm in ('field','none') else (0,(3,1.5)),zorder=4 if arm=='field' else 3)
            if b.max()>=19:ax.annotate('blocked (20)',(x[np.argmax(b)],YMAX*.97),xytext=(0,-10),textcoords='offset points',fontsize=fs(5.8),color=color,ha='center')
        lim=limit(spec);notes[scene]=dict(exposures=exposures,limit=lim,true=float(np.trapezoid(truth-.02,x)))
        ax.set_title(f'{scene.replace("_"," ")}: true excess {notes[scene]["true"]:.2f}, detour limit {lim:.2f}',fontsize=fs(6.6),pad=2)
        ax.set(xlim=(0,12),ylim=(0,YMAX),xlabel='position along the middle row (m)');ax.grid(color='#E8EAED',lw=.5)
        ax.text(-.02 if k else -.11,1.02,'ab'[k],transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom',ha='right')
    axes[0].set_ylabel('carrier belief (risk)',labelpad=1)
    handles=[Line2D([],[],color='#C62828',label='true risk')]+[Line2D([],[],color=c,ls='-' if a in ('field','none') else (0,(3,1.5)),label=l) for a,(c,l) in SHOW.items()]
    fig.legend(handles=handles,loc='lower center',ncol=3,fontsize=fs(6),handlelength=1.6,columnspacing=1.2,bbox_to_anchor=(.55,-.02))
    notes['belief_time']='carrier activation + 3 s (hazard_data.DECISION_DELAY)';save_figure(fig,'fig8_belief',sources,notes=notes);plt.close(fig)


if __name__=='__main__':main()
