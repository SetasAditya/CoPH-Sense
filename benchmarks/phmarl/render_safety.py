"""Layout-only rendering from frozen logged positions; no simulator or learner."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.patches import Circle

ARMS=['phmarl','attraction','attraction_protected','coph_frozen_protected','coph_online_protected','phmarl_protected']
LABELS=['Published phMARL','Simple attraction','Attraction + protection','Frozen CoPH + protection','Online CoPH + protection','phMARL + protection']

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path(__file__).parent/'results/native_safety_v2');args=parser.parse_args();out=args.out
    records={a:np.load(out/(a+'_seed81000.npz')) for a in ARMS}
    colors=['tab:blue','tab:orange','tab:green','tab:red'];lim=max(1.3,max(np.abs(r['positions']).max() for r in records.values())+.2)
    fig,axes=plt.subplots(2,3,figsize=(13,10));fig.subplots_adjust(top=.9,bottom=.05,hspace=.32,wspace=.2)
    fig.suptitle('Native navigation — paired fresh seed 81000\nAll methods use identical worlds, timestamps and reward',fontsize=13,y=.98)
    panels=[]
    for ax,arm,label in zip(axes.flat,ARMS,LABELS):
        r=records[arm];ax.set(xlim=(-lim,lim),ylim=(-lim,lim),aspect='equal');ax.tick_params(labelsize=8)
        ax.scatter(*r['landmarks'].T,c='black',marker='x',s=40);cs=[];ls=[]
        for i in range(4):
            c=Circle(r['positions'][0,i],.15,color=colors[i],alpha=.5);ax.add_patch(c);cs.append(c)
            ls.append(ax.plot([],[],color=colors[i],lw=1)[0])
        panels.append((ax,r,cs,ls,label))
    def update(t):
        for ax,r,cs,ls,label in panels:
            for i in range(4):cs[i].center=r['positions'][t,i];ls[i].set_data(r['positions'][:t+1,i,0],r['positions'][:t+1,i,1])
            ax.set_title('%s\nStep %d / 400 | overlap pair-steps: %d'%(label,t,r['collisions'][:t].sum()),fontsize=10,pad=8)
    FuncAnimation(fig,update,frames=range(0,401,4),interval=80).save(out/'comparison_seed81000_clean.gif',writer=PillowWriter(fps=12))
    update(400);fig.savefig(out/'comparison_seed81000_clean.png',dpi=140);plt.close(fig)

if __name__=='__main__':main()
