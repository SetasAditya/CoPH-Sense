"""Independent tiny-scene checks for communication-pilot accounting corrections."""
import copy
import unittest
from unittest.mock import patch

import numpy as np

from coph_stagewise import communication_navigation as navigation
from coph_stagewise.cooperative import CommunicationConfig
from coph_stagewise.crossing_demo import configuration


def fixture(heterogeneous=True):
    spec=dict(name='accounting_tiny_corridor',seed=812345,
        initials=[[0.,0.,0.,0.],[0.,.1,0.,0.]],goals=[[3.,0.],[3.,0.]],
        initial=[0.,0.,0.,0.],goal=[3.,0.],start_seconds=[0.,0.],steps=8,
        damping=.85,obstacles=[],patches=[[.6,0.,.5,.5,.4]],bounds=[-1.,4.,-1.,1.],
        dynamic=False,network_nodes=[[0.,0.],[3.,0.]],network_edges=[[0,1]],
        corridor_radius=.85)
    if heterogeneous:spec.update(risk_weights=[.25,6.],unknown_risk=.02)
    return spec


def analytic_risk(q,spec):
    value=.02
    for x,y,sx,sy,amplitude in spec['patches']:
        value+=amplitude*np.exp(-.5*(((q[0]-x)/sx)**2+((q[1]-y)/sy)**2))
    return float(value)


class CommunicationAccountingTests(unittest.TestCase):
    def assert_accounting(self,result,expected_weights):
        dt=result['config']['dt']
        total_components=0.
        total_recorded=0.
        for i,(agent,trajectory) in enumerate(zip(result['agents'],result['trajectories'])):
            weight=expected_weights[i]
            self.assertEqual(agent['risk_weight'],weight)
            self.assertTrue(trajectory)
            raw=0.
            sums=dict(time=0.,risk=0.,effort=0.)
            for step in trajectory:
                self.assertEqual(set(step['components']),{'time','risk','effort'})
                exposure=dt*analytic_risk(step['after'][:2],result['spec'])
                self.assertAlmostEqual(step['raw_risk'],exposure,places=14)
                self.assertAlmostEqual(step['components']['risk'],4*weight*exposure,places=14)
                self.assertAlmostEqual(step['components']['time'],dt,places=14)
                self.assertAlmostEqual(step['components']['effort'],dt*.03*float(np.dot(step['u'],step['u'])),places=14)
                self.assertAlmostEqual(step['cost'],sum(step['components'].values()),places=13)
                raw+=step['raw_risk']
                for key,value in step['components'].items():sums[key]+=value
            self.assertGreater(raw,0.)
            self.assertAlmostEqual(agent['raw_risk'],raw,places=14)
            self.assertAlmostEqual(agent['components']['risk'],4*weight*raw,places=13)
            for key,value in sums.items():self.assertAlmostEqual(agent['components'][key],value,places=13)
            self.assertEqual(agent['components']['failure'],0. if agent['success'] else 40.)
            self.assertAlmostEqual(agent['time_risk_cost'],sums['time']+sums['risk'],places=13)
            self.assertAlmostEqual(agent['cost'],sum(agent['components'].values()),places=13)
            self.assertAlmostEqual(agent['cost'],sum(step['cost'] for step in trajectory)+agent['components']['failure'],places=12)
            total_components+=sum(agent['components'].values())
            total_recorded+=sum(step['cost'] for step in trajectory)+agent['components']['failure']
        self.assertAlmostEqual(total_components,total_recorded,places=12)
        self.assertAlmostEqual(sum(a['time_risk_cost'] for a in result['agents']),
            sum(a['components']['time']+a['components']['risk'] for a in result['agents']),places=13)

    def test_factory_honors_heterogeneous_and_default_context(self):
        for heterogeneous,weights,unknown in ((True,(.25,6.),.02),(False,(1.,1.),1.)):
            spec=fixture(heterogeneous);make=navigation.factory(spec);cfg=configuration(spec['steps'])
            for i in (0,1):
                agent=make(cfg,np.asarray(spec['initials'][i]),spec['goals'][i],str(i),CommunicationConfig())
                self.assertEqual(agent.navigation.get('risk_weight',1.),weights[i])
                self.assertEqual(agent.navigation.get('unknown_risk',1.),unknown)

    def test_corrected_heterogeneous_raw_weighted_and_complete_costs(self):
        result=navigation.execute(fixture(),'disabled','rotation')
        self.assertEqual(result['accounting_mode'],'corrected')
        self.assert_accounting(result,(.25,6.))

    def test_missing_weights_keep_unit_weight_accounting(self):
        result=navigation.execute(fixture(False),'disabled','rotation')
        self.assert_accounting(result,(1.,1.))

    def test_legacy_and_corrected_preserve_physics_learning_and_transport(self):
        spec=fixture();original_factory=navigation.factory
        def run(mode):
            captured=[]
            def recording_factory(s):
                make=original_factory(s)
                def record(*args,**kwargs):
                    agent=make(*args,**kwargs);captured.append(agent);return agent
                return record
            with patch.object(navigation,'factory',recording_factory):
                result=navigation.execute(copy.deepcopy(spec),'reciprocal_fixed','receiver_relevant',
                    byte_budget=30000,accounting_mode=mode)
            models=[dict(version=a.version,damping=a.damping,policy=a.policy.tolist(),
                stages=[dict(id=s.id,version=s.version,weights=s.weights.tolist()) for s in a.stages]) for a in captured]
            return result,models
        corrected,corrected_models=run('corrected');legacy,legacy_models=run('legacy')
        self.assert_accounting(corrected,(.25,6.));self.assert_accounting(legacy,(1.,1.))
        self.assertEqual(corrected_models,legacy_models)
        for key in ('samples','observations','geometry_events','receive_update_slots','pending_packets'):
            self.assertEqual(corrected[key],legacy[key])
        for left,right in zip(corrected['trajectories'],legacy['trajectories']):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):
                for key in ('time','tick','x','u','nominal','after','version','stage','energy','raw_risk'):
                    self.assertEqual(a[key],b[key])
        for left,right in zip(corrected['updates'],legacy['updates']):
            self.assertEqual(len(left),len(right))
            for a,b in zip(left,right):
                for key in ('time','tick','version_before','version_after','packets'):self.assertEqual(a[key],b[key])
                for key in ('accepted','reason','version_after','damping','policy','counterfactual_steps',
                            'field_validation_before','field_validation_after','plant_validation_before','plant_validation_after'):
                    self.assertEqual(a['diagnostic'][key],b['diagnostic'][key])
        def wire_events(result):
            return [{k:e[k] for k in ('kind','reason','packet','sender','receiver','time','tick') if k in e}
                    for e in result['events']]
        self.assertEqual(wire_events(corrected),wire_events(legacy))
        self.assertTrue(any(e['kind']=='sent' for e in corrected['events']))
        for a,b in zip(corrected['agents'],legacy['agents']):
            self.assertEqual(a['raw_risk'],b['raw_risk'])
            self.assertNotEqual(a['components']['risk'],b['components']['risk'])
            for key in ('status','payload','acknowledgment'):self.assertEqual(a['bytes'][key],b['bytes'][key])

    def test_per_agent_ceiling_includes_status_payload_and_ack(self):
        for selector in ('rotation','receiver_relevant'):
            with self.assertRaises(ValueError):
                navigation.execute(fixture(),'reciprocal_fixed',selector,byte_budget=0)
            for budget in (1,2000):
                result=navigation.execute(fixture(),'reciprocal_fixed',selector,byte_budget=budget)
                self.assertEqual(result['communication']['byte_budget'],budget)
                for agent in result['agents']:
                    charged=sum(agent['bytes'][k] for k in ('status','payload','acknowledgment'))
                    self.assertGreaterEqual(charged,0)
                    self.assertLessEqual(charged,budget)
                    if budget==1:self.assertEqual(charged,0)
                self.assert_accounting(result,(.25,6.))


if __name__=='__main__':unittest.main()
