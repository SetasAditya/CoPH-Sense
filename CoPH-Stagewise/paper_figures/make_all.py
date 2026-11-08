"""Regenerate every paper figure. Pass newer results directories as they exist."""
import argparse
from pathlib import Path
from . import fig1_teaser,fig2_cascade,fig3_hamiltonian,fig4_superadditivity,fig5_generalization,fig6_delay,fig7_terrain,fig8_belief
from .style import ROOT


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--crossing',type=Path,default=ROOT/'results'/'crossing_traverses_v1')
    ap.add_argument('--family',type=Path,nargs='*',default=[],help='complex-maze results directories')
    ap.add_argument('--delay',type=Path,nargs='*',default=[],help='delay-sweep results directories')
    ap.add_argument('--terrain',type=Path,nargs='*',default=[],help='graded-hazard pilot root')
    args=ap.parse_args(argv);c=['--results',str(args.crossing)]
    fig1_teaser.main(c);fig2_cascade.main(c);fig3_hamiltonian.main(c)
    fig4_superadditivity.main(['--results',str(args.crossing)])
    fig5_generalization.main(['--results',str(args.crossing),*map(str,args.family)]);fig6_delay.main(['--results',*([str(args.crossing)] if args.delay else []),*map(str,args.delay)]);fig7_terrain.main(['--results',*map(str,args.terrain)]);fig8_belief.main(['--results',*map(str,args.terrain)])
    print('figures written to',ROOT/'paper_figures'/'output')


if __name__=='__main__':main()
