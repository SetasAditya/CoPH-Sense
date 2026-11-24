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
 for i,(v,c) in enumerate(zip(vals,cov)):ax.text(v+5,i,f'{v:.1f}\n{c}/{n}',va='center',ha='left',fontsize=11,color='white',weight='bold')
fig.suptitle('Online CoPH versus the same pretrained phMARL checkpoint',fontsize=17,weight='bold')
fig.text(.5,.035,'Same 4-agent task, task allocation, targets and protection within each cohort. Higher return is better.\nSecond bar label = episodes with all four landmarks covered. Each arm records one contact per cohort. Cohorts are not pooled.',ha='center',fontsize=10)
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
body='<p>The sections below explain each experiment and compare the methods on the same task. <strong>Blue rows mark the best result on the main measure, including ties.</strong> Results from different studies are kept separate. We have not yet compared CoPH with DistGP or the exposure planner.</p>'
body+=section('dynamics','1. Adapting when drag changes','Four agents must cover four landmarks in VMAS simple_spread. At 20 s, drag changes from 0.25 and the landmarks move. We test each direction on 30 starts, with the same assignments, target selection and protection for every controller. The table reports reward summed after the change. Higher return is better: −73 is better than −106.',figure('motivation_adaptation','The fixed model starts with the correct drag value and performs better before the change. Afterward, it keeps the old value. Online CoPH gains +33.3 return [30.6, 36.0] after the drag increase and +29.7 [24.8, 34.8] after the decrease, relative to that fixed model. Brackets show paired 95% intervals; bars show mean return.')+''.join('<h3>Drag 0.25 → '+cond+'</h3>'+table(['Controller / model','Post-change return ↑','All landmarks covered ↑','New contact onsets ↓'],[[labels[i].replace('\n',' '),f'{drag[cond]["arms"][k]["phase2_return"]:.1f}',f'{drag[cond]["arms"][k]["full_phase2"]}/30',drag[cond]['arms'][k]['new_contacts']] for i,k in enumerate(keys)],{3}) for cond in ['0.45','0.1']),'CoPH has higher return after the change than every tested fixed model, in both directions. This experiment tests dynamics learning, not communication or replay.')
body+=section('native','2. Comparing with phMARL on landmark coverage','Four agents must cover four landmarks. We compare one published phMARL checkpoint with CoPH, using the same assignments, static targets and protection. The 50-start and 100-start tests use separate groups of initial conditions.',figure('comparison_native','Mean return and the number of episodes in which all four landmarks are covered. The 50-start return gain is +112.4 [102.4, 122.8]; the 100-start gain is +105.3 [97.9, 112.6]. The brackets are paired 95% intervals for the difference between methods.')+table(['Cohort','Controller','Mean return ↑','Full coverage ↑','New contacts ↓'],[[f'{n} starts',name,f'{data[k]["mean_return"]:.1f}',f'{data[k][cov]}/{n}',int(data[k]['new_contacts'])] for n,data,cov,ks in [(50,native,'full_coverage',['phmarl_assigned_static_protected','coph_online_static_protected']),(100,fresh,'coverage',['final/phmarl_assigned_static_protected','final/coph_online_static_protected'])] for name,k in zip(['Matched phMARL','CoPH online'],ks)],{1,3}),'CoPH covers all landmarks more often and earns higher return than this checkpoint. The result applies to the tested task and checkpoint.')
body+=section('uncertainty','3. Choosing paths from a few measurements','Each method receives the same few measurements and chooses between the same two paths. We test 400 path pairs in each of 12 saved-state groups. A choice is labeled certified when 95% of samples from the method’s stated uncertainty agree. This label is not a safety guarantee.',figure('comparison_representation','The field makes fewer costly choices and gives uncertainty intervals closer to their stated coverage. It also certifies fewer choices. Regret is the extra true cost of choosing the more expensive path. These tests use saved states, not complete navigation runs.')+table(['Representation','Mean regret ↓','Correct choice ↑','Choices certified','Wrong among certified ↓','90% interval coverage (target 90%)'],[[name,f'{regret[k]["mean_regret"]:.3f}',f'{100*regret[k]["accuracy"]:.1f}%',f'{100*cert[k]["certified_share"]:.1f}%',f'{100*cert[k]["wrong_when_certified"]:.1f}%',f'{100*cert[k]["calibration90"]:.1f}%'] for k,name in zip(rkeys,['Raw readings','Graded costmap','CoPH field'])],{2}),'The field has lower regret in all 12 groups. The gain is about 1% of path cost. DistGP is not part of this test.')
body+=section('hazards','4. Sharing hazards with a more cautious teammate','In four hazard scenes, the scout has risk weight 0.25 and the carrier has weight 6. The carrier starts 25 s later. Both can cross the hazards, but the carrier pays a larger risk penalty. All 64 agent missions finish without collisions. Cost is travel time plus weighted risk. Excess cost is measured against the best tested method in each scene; lower is better.',figure('comparison_hazards','The scout’s binary advice can be unsuitable for the carrier. Hazard values let the carrier use its own risk weight. The learned field and raw readings perform equally well in this pilot.')+table(['Shared information','Worst-case carrier excess cost ↓','Interpretation'],[['Learned field','4.6','Carrier uses the hazard values with its own risk weight'],['Raw readings','4.6','Matches learned-field navigation'],['Field, receiver fitting disabled','4.6','Same result without fitting the received field'],["Scout’s avoid/free decision",'25.3',"The advice reflects the scout’s risk weight"],['No communication','68.1','Carrier must encounter the terrain itself']],{0,1,2}),'Sending hazard values works better than binary advice or no communication. Raw readings tie the learned field. Receiver fitting does not change the navigation outcome here.')
body+=section('reciprocity','5. Sending warnings in both directions','Two agents know the corridor network but not which corridors are blocked. We compare two-way warnings with no communication on the same layout. The animation uses seed 76000.',figure('reciprocity','Solid paths show two-way communication; dashed paths show no communication. A warning changes one agent’s route, allowing it to find a second blockage sooner and warn its teammate. This is one recorded example.')+'<p>In this layout, reciprocal exchange saves <strong>27.1 m of team detour</strong>. The summary does not include total detour distances for each method, so only the measured difference is reported here.</p>','Warnings in both directions save travel in this layout. How much they help depends on the network and when the warnings arrive. No external planner is compared here.')
body+=section('encounters','6. Getting through a shared crossing','Four agents swap places through the same center. With the old protection layer, they can all wait for one another. The final layer uses a fixed priority rule so they can take turns. We keep the controller unchanged and test 30 starts per encounter type.',table(['Four-agent swap','Completed ↑','What changes'],[['CoPH + old layer','0/30','All agents can remain waiting'],['CoPH + final layer','30/30','A priority rule lets agents take turns']],{1})+'<p>With the final layer, both CoPH and the attraction controller finish all 30 starts in each tested encounter type, without contacts. The protection rule produces this improvement. Taking turns adds delay: the reported median is 140 crossing steps versus 81.</p>','The rule clears the tested deadlocks. CoPH has not learned to yield, and the attraction controller also benefits from the same rule.')
page='<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Comparative evidence · CoPH</title><link rel="stylesheet" href="style.css"><header class="topbar"><a class="brand" href="index.html">CoPH</a><a href="index.html#scenarios">Back to scenarios</a></header><main><section><p class="eyebrow">TASK · BASELINE · METRIC · RESULT</p><h1>Where CoPH helps.</h1>'+body+'<p><a href="records.html">Recorded signals and episode outcomes</a></p></section></main></html>'
# Version assets so browsers do not reuse plots or styles from earlier revisions.
import re
page=re.sub(r'((?:src|href)=")((?:assets/[^" ]+\.(?:svg|png))|style\.css)(")',lambda m:m[1]+m[2]+'?v='+hashlib.sha256((DIST/m[2]).read_bytes()).hexdigest()[:12]+m[3],page)
page=page.replace('<h1>Where CoPH helps.</h1>','<h1>Where CoPH helps.</h1><p class="fine">Updated 24 November 2026.</p>')
(DIST/'results.html').write_text(page)
(DIST/'assets/comparison_sources.json').write_text(json.dumps({'summary_sha256':used,'new_experiments':False,'figures':['motivation_adaptation','comparison_native','comparison_representation','comparison_hazards']},indent=2)+'\n')
print('Built comparative figures and contextual tables from five preserved summaries.')
