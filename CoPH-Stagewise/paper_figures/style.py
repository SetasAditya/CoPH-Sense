"""Paper styling. Mirrors rq_visualizations/common.py so both papers' figures read
alike, but is copied rather than imported to keep CoPH-Stagewise self-contained."""
import hashlib
import os
import json
from pathlib import Path
import matplotlib as mpl

ROOT=Path(__file__).resolve().parents[1]
OUTPUT=ROOT/'paper_figures'/'output'

# Single-column text width (in) and the point size text should have on the page.
TEXT_WIDTH_IN=5.5
TARGET_PT=7.5

A,B='#0072B2','#D55E00'
AGENT=(A,B)
MESSAGE='#8E24AA'
DOOR='#C62828'
HIDDEN='#8E949C'
WALL='#2F3640'
FLOOR='#ECEEF0'
MUTED='#9AA1A9'
ARMS=('disabled','one_way_01','one_way_10','reciprocal_fixed')
ARM_SHORT=dict(disabled='none',one_way_01='A→B only',one_way_10='B→A only',reciprocal_fixed='reciprocal')
ARM_COLOR=dict(disabled='#7A7F86',one_way_01='#B39DDB',one_way_10='#9575CD',reciprocal_fixed=MESSAGE)

_SCALE=1.


def setup_style(fig_width_in=TEXT_WIDTH_IN,*,frac=1.,target_pt=TARGET_PT):
    """Style so text lands at target_pt once the figure is drawn at frac*\\textwidth."""
    global _SCALE
    scale=TEXT_WIDTH_IN*frac/float(fig_width_in);base=target_pt/scale;_SCALE=base/target_pt
    mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':base,'axes.titlesize':base,'axes.labelsize':base,
        'xtick.labelsize':base*.92,'ytick.labelsize':base*.92,'legend.fontsize':base*.92,'figure.titlesize':base*1.1,
        'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'lines.linewidth':1.4,
        'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.5,'ytick.major.size':2.5,
        'figure.dpi':140,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42})
    return scale


def fs(pt):
    """A hand-placed size, in points as it should appear on the page."""
    return pt*_SCALE


def rendered_pt(fig_width_in,pt,*,frac=1.):
    return pt*TEXT_WIDTH_IN*frac/float(fig_width_in)


def panel_label(ax,label,x=-.02,y=1.02):
    ax.text(x,y,label,transform=ax.transAxes,fontsize=fs(8.5),weight='bold',va='bottom',ha='right')


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as handle:
        for block in iter(lambda:handle.read(1<<20),b''):digest.update(block)
    return digest.hexdigest()


def save_figure(fig,stem,inputs,output=OUTPUT,notes=None):
    # No bbox cropping: it changes the on-page scale the font sizing assumes.
    output.mkdir(parents=True,exist_ok=True)
    fig.savefig(output/f'{stem}.pdf');fig.savefig(output/f'{stem}.png')
    record=dict(figure=stem,width_in=fig.get_figwidth(),height_in=fig.get_figheight(),
                inputs=[dict(path=os.path.relpath(Path(p).resolve(),ROOT),sha256=sha256(p)) for p in inputs],notes=notes or {})
    (output/f'{stem}.provenance.json').write_text(json.dumps(record,indent=2)+'\n')
