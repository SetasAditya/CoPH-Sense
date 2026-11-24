"""Self-contained comparison figures and tables from preserved study summaries."""
from pathlib import Path
import json
import hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from html import escape

ROOT=Path(__file__).resolve().parent; DIST=ROOT/'dist'; ASSETS=DIST/'assets'
SOURCES=ROOT.parent/'docs/results'
used={}
def load(study):
 p=SOURCES/study/'summary.json';used[str(p.relative_to(ROOT.parent))]=hashlib.sha256(p.read_bytes()).hexdigest();return json.loads(p.read_text())
drag=load('native_drag_shift_v1')['conditions'];native=load('native_heldout_v1')['summary'];fresh=load('native_final_layer_v1')['native'];regret=load('reconstruction_decision_v1')['all'];cert=load('route_certificate_v1')
BLUE='#2474c8';ORANGE='#d87928';GRAY='#8996a3';TEAL='#007f7b';INK='#152d43'
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':INK,'text.color':INK,'figure.facecolor':'white','savefig.facecolor':'white'})
keys=['phmarl_assigned_static_protected','coph_frozen_static_protected','coph_known_static_protected','coph_online_static_protected']
labels=['Matched\nphMARL','Frozen model\n(drag 0.058)','Initially correct\nfixed model (0.25)','CoPH\nonline'];colors=[ORANGE,GRAY,TEAL,BLUE]
def save(fig,name):
 fig.savefig(ASSETS/(name+'.svg'),bbox_inches='tight');fig.savefig(ASSETS/(name+'.png'),dpi=180,bbox_inches='tight');plt.close(fig)
# Motivation explicitly separates a schematic mechanism from measured outcomes.
fig=plt.figure(figsize=(13,8.5));gs=fig.add_gridspec(2,2,height_ratios=[.65,1],hspace=.57,wspace=.65)
a=fig.add_subplot(gs[0,:]);a.set_axis_off();a.set_xlim(0,1);a.set_ylim(0,1)
boxes=[('1  Adequate model','True drag = model drag = 0.25\nFour agents cover four landmarks'),('2  Surprise at 20 s','Drag changes; landmarks move\nThe team must travel again'),('3  Revise and act','Executed transitions → drag estimate\nUpdated model → subsequent control')]
for i,(title,desc) in enumerate(boxes):
 x=.015+i*.335;a.add_patch(FancyBboxPatch((x,.1),.30,.8,boxstyle='round,pad=.01',facecolor='#f3f7fa',edgecolor=BLUE if i==2 else '#ccd9e3'))
 a.text(x+.015,.72,title,weight='bold',fontsize=12);a.text(x+.015,.47,desc,fontsize=10.3,va='top',linespacing=1.6)
 if i<2:a.annotate('',(x+.33,.5),(x+.302,.5),arrowprops={'arrowstyle':'->','color':INK,'lw':2})
for j,condition in enumerate(['0.45','0.1']):
 ax=fig.add_subplot(gs[1,j]);arms=drag[condition]['arms'];means=[arms[k]['phase2_return'] for k in keys]
 ax.barh(range(4),means,color=colors);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlim(-175,0);ax.set_xlabel('Mean post-change return ↑ (less negative is better)');ax.set_title('Drag 0.25 → '+condition+' | 30 paired starts',weight='bold',pad=12)
 for i,v in enumerate(means):ax.text(v+3,i,f'{v:.1f}',va='center',ha='left',fontsize=11,color='white',weight='bold')
 ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
fig.suptitle('Why adapt online? A correct model can become stale.',fontsize=20,weight='bold',y=1.02)
fig.text(.5,-.035,'Measured benchmark: 4-agent VMAS simple_spread; 400 × 0.1 s steps; change at step 200.\nAll arms share allocation, targets and protection. Top: mechanism schematic. Bottom: archived means; no error bars inferred.',ha='center',fontsize=10.5)
save(fig,'motivation_adaptation')
# Baseline comparison across separate cohorts, never pooled.
fig,axes=plt.subplots(1,2,figsize=(12,5));fig.subplots_adjust(wspace=.35,bottom=.22,top=.78)
for ax,title,data,n in [(axes[0],'Original held-out cohort | 50 starts',native,50),(axes[1],'Fresh cohort, final protection | 100 starts',fresh,100)]:
 ks=['phmarl_assigned_static_protected','coph_online_static_protected'];prefix='final/' if n==100 else ''; vals=[data[prefix+k]['mean_return'] for k in ks];cov=[data[prefix+k]['coverage' if n==100 else 'full_coverage'] for k in ks]
 ax.barh([0,1],vals,color=[ORANGE,BLUE]);ax.set_yticks([0,1],['Matched phMARL','CoPH online']);ax.invert_yaxis();ax.set_xlim(-235,0);ax.set_xlabel('Mean episode return ↑');ax.set_title(title,fontsize=12)
 for i,(v,c) in enumerate(zip(vals,cov)):ax.text(v+5,i,f'{v:.1f}\nCoverage {c}/{n}',va='center',ha='left',fontsize=11,color='white',weight='bold')
fig.suptitle('Online CoPH versus the same pretrained phMARL checkpoint',fontsize=17,weight='bold')
fig.text(.5,.035,'Same 4-agent task, task allocation, targets and protection within each cohort. Higher return is better.\nCoverage = all four landmarks covered. Each compared arm records one contact in each cohort. Cohorts are not pooled.',ha='center',fontsize=10)
save(fig,'comparison_native')
# Representation comparison includes both uncertainty selectivity and error.
fig,axes=plt.subplots(1,3,figsize=(13,5));fig.subplots_adjust(wspace=.34,bottom=.24,top=.78)
rkeys=['raw','costmap','field'];rlabs=['Raw\nreadings','Graded\ncostmap','CoPH\nfield'];cs=[GRAY,ORANGE,BLUE]
metrics=[([regret[k]['mean_regret'] for k in rkeys],'Mean route regret ↓','{:.3f}'),([100*cert[k]['wrong_when_certified'] for k in rkeys],'Wrong among certified choices (%) ↓','{:.1f}%'),([100*cert[k]['calibration90'] for k in rkeys],'90% interval coverage (%)','{:.1f}%')]
for ax,(vals,title,fmt) in zip(axes,metrics):
 ax.bar(range(3),vals,color=cs);ax.set_xticks(range(3),rlabs);ax.set_title(title,fontsize=11.5);ax.set_ylim(0,max(vals)*1.25)
 for i,v in enumerate(vals):ax.text(i,v+max(vals)*.035,fmt.format(v),ha='center',weight='bold' if i==2 else 'normal')
 if ax is axes[2]:ax.axhline(90,color=TEAL,ls='--',lw=1);ax.text(2.5,90,'90% target',ha='right',va='bottom',fontsize=9)
fig.suptitle('Same evidence, same candidate paths: representation affects decisions',fontsize=17,weight='bold')
fig.text(.5,.025,'12 saved-state groups × 400 path pairs; no closed-loop navigation. Choices certified: raw 71.0%, costmap 69.4%, field 65.7%.\nRegret = chosen path’s true cost − cheaper candidate’s true cost. Dependent queries are not 4,800 independent trials.',ha='center',fontsize=10)
save(fig,'comparison_representation')

# Communication motivation: graded evidence versus a sender's binary advice.
fig,ax=plt.subplots(figsize=(11,5));fig.subplots_adjust(left=.31,bottom=.27,top=.78)
hnames=['No communication',"Scout’s avoid/free advice",'Raw readings','CoPH learned field']
hvalues=[68.1,25.3,4.6,4.6]
ax.barh(range(4),hvalues,color=[GRAY,ORANGE,TEAL,BLUE]);ax.set_yticks(range(4),hnames);ax.invert_yaxis();ax.set_xlim(0,78)
for i,v in enumerate(hvalues):ax.text(v+1,i,str(v),va='center',weight='bold' if i>=2 else 'normal')
ax.set_xlabel('Worst-case carrier excess cost ↓ (time + receiver-weighted risk)')
fig.suptitle('Share the hazard evidence—not the scout’s preference',fontsize=18,weight='bold')
fig.text(.5,.035,'Risk-tolerant scout (weight 0.25); risk-averse carrier (weight 6), starting 25 s later. Four traversable-hazard scenes.\nAll 64 agent missions complete without collisions. Excess is relative to the best tested arm per scene. Field and raw readings tie.',ha='center',fontsize=10)
save(fig,'comparison_hazards')

def table(headers,rows,best=None):
 out='<div class="table-wrap"><table><thead><tr>'+''.join('<th scope="col">'+h+'</th>' for h in headers)+'</tr></thead><tbody>'
 for i,row in enumerate(rows):
  out+='<tr'+(' class="best-row"' if best is not None and i in best else '')+'>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>'
 return out+'</tbody></table></div>'
def figure(name,caption):
 ext='.svg' if (ASSETS/(name+'.svg')).exists() else '.png'
 return '<figure><a href="assets/'+name+ext+'"><img class="result-figure" src="assets/'+name+ext+'" alt="'+escape(caption)+'" loading="lazy"></a><figcaption>'+caption+'</figcaption></figure>'
def section(key,title,setup,body,takeaway):return '<section id="'+key+'"><h2>'+title+'</h2><p class="study-context">'+setup+'</p>'+body+'<p class="takeaway"><strong>Takeaway.</strong> '+takeaway+'</p></section>'
body='<p>Each comparison below states the task, shared conditions, metric and baseline. <strong>Blue rows identify the best measured result for the stated primary metric; ties are retained.</strong> Separate studies are not pooled. DistGP and exposure-planning comparisons remain unrun.</p>'
body+=section('dynamics','1. Unexpected dynamics: when online identification is needed','Task: four agents cover four landmarks in VMAS simple_spread. At 20 s, drag changes from 0.25 and the landmarks move. Each direction uses 30 paired starts. All arms share assignment, target selection and protection. Return is the sum of task rewards; higher (less negative) is better. Only the post-change phase is compared below.',figure('motivation_adaptation','An initially correct model performs better before the intervention, but becomes stale afterward. Online CoPH gains +33.3 return [30.6, 36.0] after the drag increase and +29.7 [24.8, 34.8] after the decrease, relative to that fixed model. Brackets are archived paired 95% intervals; bars show means.')+''.join('<h3>Drag 0.25 → '+cond+'</h3>'+table(['Controller / model','Post-change return ↑','All landmarks covered ↑','New contact onsets ↓'],[[labels[i].replace('\n',' '),f'{drag[cond]["arms"][k]["phase2_return"]:.1f}',f'{drag[cond]["arms"][k]["full_phase2"]}/30',drag[cond]['arms'][k]['new_contacts']] for i,k in enumerate(keys)],{3}) for cond in ['0.45','0.1']),'Online adaptation beats every tested fixed model in post-change return in both directions. This supports dynamics adaptation on this task; it does not establish a communication or replay benefit.')
body+=section('native','2. Native task: comparison with matched phMARL','Task: the same four-agent landmark coverage benchmark. The external baseline is one published pretrained phMARL checkpoint, given the same allocation, static targets and protection as CoPH. These are two separate held-out cohorts.',figure('comparison_native','Absolute mean returns and full-coverage counts for matched phMARL and online CoPH. The 50-start return gain is +112.4 [102.4, 122.8]; the 100-start gain is +105.3 [97.9, 112.6]. These are paired archived 95% intervals, not bar error estimates.')+table(['Cohort','Controller','Mean return ↑','Full coverage ↑','New contacts ↓'],[[f'{n} starts',name,f'{data[k]["mean_return"]:.1f}',f'{data[k][cov]}/{n}',int(data[k]['new_contacts'])] for n,data,cov,ks in [(50,native,'full_coverage',['phmarl_assigned_static_protected','coph_online_static_protected']),(100,fresh,'coverage',['final/phmarl_assigned_static_protected','final/coph_online_static_protected'])] for name,k in zip(['Matched phMARL','CoPH online'],ks)],{1,3}),'CoPH improves coverage and return against this matched checkpoint. The comparison does not establish superiority over all phMARL training runs, tasks or MARL methods.')
body+=section('uncertainty','3. Sparse evidence: field versus raw readings and costmap','Task: choose the lower-cost candidate path from identical sparse evidence. Twelve saved-state groups contain 400 path pairs each. All representations score the same candidates. A choice is certified when it agrees across 95% of samples from that representation’s stated uncertainty.',figure('comparison_representation','The field reduces mean regret and has better empirical interval coverage, while certifying fewer choices. Nominal 90% coverage is a calibration target, not a quantity to maximize without limit. These are saved-state decisions, not navigation episodes.')+table(['Representation','Mean regret ↓','Correct choice ↑','Choices certified','Wrong among certified ↓','90% interval coverage (target 90%)'],[[name,f'{regret[k]["mean_regret"]:.3f}',f'{100*regret[k]["accuracy"]:.1f}%',f'{100*cert[k]["certified_share"]:.1f}%',f'{100*cert[k]["wrong_when_certified"]:.1f}%',f'{100*cert[k]["calibration90"]:.1f}%'] for k,name in zip(rkeys,['Raw readings','Graded costmap','CoPH field'])],{2}),'The field has lower regret in all 12 groups; the absolute advantage is small (about 1% of path cost). DistGP was not evaluated in this comparison.')
body+=section('hazards','4. Different preferences: what should a scout send?','Original four-scene pilot: a risk-tolerant scout (weight 0.25) precedes a risk-averse carrier (weight 6) by 25 s. Hazards are traversable. All 64 agent missions complete without collisions. Cost is travel time plus receiver-weighted risk. Worst-case excess is relative to the best tested arm in each scene; lower is better.',figure('comparison_hazards','Different objectives make a sender’s binary advice insufficient. Graded field summaries let the carrier price the same hazard using its own risk weight. Raw readings achieve the same result in this pilot; the figure does not claim a field advantage over raw evidence.')+table(['Shared information','Worst-case carrier excess cost ↓','Interpretation'],[['Learned field','4.6','Graded evidence; carrier applies its own risk preference'],['Raw readings','4.6','Matches learned-field navigation'],['Field, receiver fitting disabled','4.6','Shows the benefit does not require receiver reconstruction here'],["Scout’s avoid/free decision",'25.3',"Applies the scout’s preference before transmission"],['No communication','68.1','Carrier learns through its own visit']],{0,1,2}),'Graded evidence beats the tested binary advice and no communication. The learned field ties raw readings, and receiver fitting adds no navigation benefit in this pilot. This table is not evidence of universal field superiority.')
body+=section('reciprocity','5. Reciprocal discovery: one warning changes the next discovery','Displayed case: two independent agents traverse a surveyed network with hidden closures. Compare reciprocal exchange with no communication on the same controlled layout. Geometry warnings may change route and discovery order. The animation uses seed 76000.',figure('reciprocity','A logged frame from the controlled crossing: solid and dashed trajectories show reciprocal exchange and no communication. Model surfaces and packet events come from recorded states; the image is an example, not an aggregate performance plot.')+'<p>In this layout, reciprocal exchange saves <strong>27.1 m of team detour</strong>. Absolute baseline detour totals are not included in the compact presentation summary, so this section makes no broader aggregate comparison.</p>','Reciprocal discovery can reduce wasted travel in a controlled layout. The benefit depends on topology and message timing; this is not a result against an external planning baseline.')
body+=section('encounters','6. Symmetric conflicts: execution-layer liveness','Task: four agents swap goals through a shared center. Compare the old and final protection layers, holding the controller fixed. The final layer includes deterministic priority; it is programmed rather than learned. Thirty starts are tested per encounter family.',table(['Four-agent swap','Completed ↑','What changes'],[['CoPH + old layer','0/30','Symmetric conflict can persist'],['CoPH + final layer','30/30','Priority resolves the conflict']],{1})+'<p>Across the tested families, the final layer completes 30/30 starts per family for both CoPH and the attraction controller, with zero contacts. Thus this liveness gain belongs to the shared execution mechanism. Priority adds delay: median crossing steps are 140 versus 81 in the reported comparison.</p>','The final layer resolves these tested conflicts; it is not evidence that CoPH learned a yielding strategy or outperformed attraction control on liveness.')
page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Comparative evidence · CoPH</title><link rel="stylesheet" href="style.css"><header class="topbar"><a class="brand" href="index.html">CoPH</a><a href="index.html#scenarios">Back to scenarios</a></header><main><section><p class="eyebrow">TASK · BASELINE · METRIC · RESULT</p><h1>Where CoPH helps.</h1>'+body+'<p><a href="records.html">Full signal and adverse-record inventory</a></p></section></main></html>'
(DIST/'results.html').write_text(page)
(DIST/'assets/comparison_sources.json').write_text(json.dumps({'summary_sha256':used,'new_experiments':False,'figures':['motivation_adaptation','comparison_native','comparison_representation','comparison_hazards']},indent=2)+'\n')
print('Built comparative figures and contextual tables from five preserved summaries.')
