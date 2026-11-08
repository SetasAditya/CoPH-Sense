import copy
import json
import unittest
import numpy as np
import torch
from coph_stagewise.records import Config
from coph_stagewise.world import World
from coph_stagewise.scenarios import SCENARIOS, specification, ScenarioWorld
from coph_stagewise.diagnostics import InstrumentedLearner, snapshot, probes, probe_metrics, replay_audit
from coph_stagewise.evaluation import execute
from coph_stagewise.cli import rollout
from coph_stagewise.visualize import common_probe_error, frame, plt

class EvaluationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1)

    def test_manifest_pairing_and_long_geometry(self):
        for name in SCENARIOS:
            self.assertEqual(specification(name,10000),specification(name,10000))
        a=specification('risk_passage',3)
        for name in ('blocked_avoidance','damping_in','damping_out'):
            b=specification(name,3);self.assertEqual(a['patches'],b['patches'])
        s=specification('long_corridor',3)
        self.assertGreaterEqual(s['nominal_stage_count'],8)
        self.assertEqual(s['steps'],576)
        for name in ('risk','obstacle'):
            old=World(5,name,Config());new=ScenarioWorld(specification(name,5),Config())
            self.assertEqual(old.sense(old.initial,0),new.sense(new.initial,0))
            self.assertEqual(old._damping,new._damping)

    def test_audit_isolation_starts_horizons_and_costs(self):
        cfg=Config();w=ScenarioWorld(specification('risk',5),cfg)
        a=InstrumentedLearner(cfg,w.initial,w.goal);a.observe(w.sense(w.initial,0))
        before=copy.deepcopy(snapshot(a));count=a.counterfactual_steps
        s=a.stages[0];audit=replay_audit(a,w,s,s.weights,a.damping,a.policy,a.policy)
        self.assertEqual(before,snapshot(a));self.assertEqual(count,a.counterfactual_steps)
        self.assertEqual(audit['predicted_improvement'],0);self.assertEqual(audit['realized_improvement'],0)
        for branch in audit['results'].values():
            self.assertEqual(branch['states'][0],list(s.entry))
            self.assertEqual(len(branch['states']),cfg.replay_steps+1)
        self.assertEqual(audit['audit_steps'],4*cfg.replay_steps)
        points=probes(w.spec);metrics=probe_metrics(a,w,points)
        self.assertEqual(before,snapshot(a))
        for q,ok in zip(points,metrics['support']):self.assertEqual(ok,s.supported(q,cfg))

    def test_costs_budget_and_early_termination(self):
        spec=specification('clear',10000,'smoke');spec['steps']=17
        full=execute(spec,'smoke','full')
        current=execute(spec,'smoke','current_replay',full['summary']['counterfactual_steps'])
        for d in (full,current):
            self.assertAlmostEqual(d['summary']['mission_cost'],sum(d['summary']['components'].values()))
            self.assertEqual(len(d['observations']),len(d['trajectory'])+1)
        self.assertEqual(full['summary']['counterfactual_steps'],current['summary']['counterfactual_steps'])
        spec['goal']=[.01,0.]
        self.assertEqual(execute(spec,'smoke','frozen')['summary']['physical_steps'],1)

    def test_no_training_rule_change(self):
        spec=specification('risk',5);spec['steps']=17
        new=execute(spec,'development','full');old=rollout(Config(steps=17),5,'risk','full')
        np.testing.assert_allclose([r['x'] for r in new['trajectory']],[r['x'] for r in old['trajectory']],atol=0,rtol=0)

    def test_animation_uses_recorded_states_and_versions(self):
        spec=specification('clear',10000,'smoke');spec['steps']=2
        d=execute(spec,'smoke','full');f=frame(d,1)
        position=f.axes[0].lines[-1]
        self.assertEqual(float(position.get_xdata()[0]),d['trajectory'][1]['after'][0])
        self.assertEqual(float(position.get_ydata()[0]),d['trajectory'][1]['after'][1])
        self.assertTrue(any(f"version {d['updates'][0]['post']['version']}" in t.get_text() for ax in f.axes for t in ax.texts))
        plt.close(f)
        d['trajectory']=[];f=frame(d,0)
        self.assertTrue(all(any('unavailable' in t.get_text() for t in ax.texts) for ax in f.axes));plt.close(f)

    def test_common_masks_missing_and_legacy_render(self):
        p=dict(support=[True,False],squared_error=[2.,None],gradient_squared_error=[3.,None])
        a=dict(probe_points=[[0,0],[1,0]],final_probes=p)
        b=copy.deepcopy(a);b['final_probes']['support']=[False,True]
        self.assertEqual(common_probe_error(a,b)['count'],0)
        self.assertIsNone(common_probe_error(a,b)['left_mse'])
        b['probe_points'][0]=[9,9]
        with self.assertRaises(ValueError):common_probe_error(a,b)
        d=dict(summary=dict(seed=0,success=False),trajectory=[dict(tick=0,x=[0,0,0,0])])
        f=frame(d,0);self.assertTrue(any('unavailable' in t.get_text() for ax in f.axes for t in ax.texts));plt.close(f)

if __name__=='__main__':unittest.main()
