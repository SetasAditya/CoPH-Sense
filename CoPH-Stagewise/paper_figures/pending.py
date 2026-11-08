"""Visible placeholder for a figure whose runs have not been executed yet."""
import matplotlib.pyplot as plt
from .style import setup_style,save_figure,fs


def placeholder(stem,title,needs,height=1.8):
    setup_style(5.5);fig=plt.figure(figsize=(5.5,height));ax=fig.add_axes([.02,.05,.96,.9]);ax.axis('off')
    ax.add_patch(plt.Rectangle((0,0),1,1,transform=ax.transAxes,fill=False,ls=(0,(4,3)),ec='#9AA1A9',lw=.8))
    ax.text(.5,.62,title,ha='center',va='center',fontsize=fs(8),weight='bold',transform=ax.transAxes)
    ax.text(.5,.38,'PENDING DATA — '+needs,ha='center',va='center',fontsize=fs(6.8),color='#8B2424',transform=ax.transAxes)
    save_figure(fig,stem,[],notes=dict(pending=needs));plt.close(fig)
