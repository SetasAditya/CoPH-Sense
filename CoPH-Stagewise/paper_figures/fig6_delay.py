"""F6: saving vs message delay. The oracle curve needs no runs; measured points do."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from coph_stagewise.crossing_demo import specification
from .style import setup_style,save_figure,fs,ARM_COLOR,ROOT
from .data import results,wasted,oracle

SPEED_SOURCE=ROOT/'results'/'crossing_traverses_v1'


def cruise_speed(episodes):
    """Median logged speed while moving, used to convert delay into distance."""
    v=[np.hypot(*s['x'][2:]) for e in episodes.values() for ss in e['samples'] for s in ss]
    v=np.array(v);return float(np.median(v[v>.1]))


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,nargs='*',default=[]);args=ap.parse_args(argv)
    base,sources=results(SPEED_SOURCE);speed=cruise_speed(base);spec=specification(76000)
    delays=np.linspace(0,40,81);curves={}
    for arm in ('one_way_10','reciprocal_fixed'):
        none=oracle(spec,'disabled')['team_wasted'];curves[arm]=[none-oracle(spec,arm,delay=d,speed=speed)['team_wasted'] for d in delays]
    setup_style(5.5);fig,ax=plt.subplots(figsize=(3.4,1.9),gridspec_kw=dict(left=.15,right=.97,top=.9,bottom=.22))
    for arm,label in (('one_way_10','B→A only'),('reciprocal_fixed','reciprocal')):ax.plot(delays,curves[arm],color=ARM_COLOR[arm],lw=1.3,label=f'{label} (oracle)')
    measured=0
    for d in args.results:
        episodes,src=results(d);sources+=src;manifest=json.loads((Path(d)/'manifest.json').read_text());delay=manifest['communication']['delay_ticks']*manifest['config']['dt']
        for (seed,arm),e in episodes.items():
            if arm not in ('one_way_10','reciprocal_fixed') or (seed,'disabled') not in episodes:continue
            ax.scatter(delay,sum(wasted(episodes[(seed,'disabled')]))-sum(wasted(e)),color=ARM_COLOR[arm],edgecolor='black',lw=.3,s=fs(10),zorder=4);measured+=1
    if not measured:ax.text(.97,.95,'measured points: pending delay sweep',transform=ax.transAxes,ha='right',va='top',fontsize=fs(6),color='#8B2424')
    ax.set(xlabel='message delay (s)',ylabel='team detour saved (m)',xlim=(0,40),ylim=(0,None));ax.grid(color='#E8EAED',lw=.5)
    ax.legend(fontsize=fs(6),loc='center right',handlelength=1.4);ax.set_title(f'Value of a warning decays with delay (speed {speed:.2f} m/s)',fontsize=fs(6.8),pad=2)
    save_figure(fig,'fig6_delay',sources,notes=dict(speed=speed,delays=delays.tolist(),oracle=curves,measured_points=measured));plt.close(fig)


if __name__=='__main__':main()
