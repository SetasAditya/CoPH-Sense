import unittest
import numpy as np
from coph_terrain.algorithm import acquire_to_evidence
from coph_terrain.frozen_need_a4 import FrozenNeedA4, legal_need_features
from coph_terrain.lookahead import LookaheadSpec
from coph_terrain.lookahead_bidirectional import BidirectionalLookaheadEnv, NeedPacket
from coph_terrain.physical_audit import AuditChoice

class FrozenNeedA4Tests(unittest.TestCase):
    def source(self):
        env = BidirectionalLookaheadEnv(LookaheadSpec(980000))
        got = acquire_to_evidence(env, AuditChoice("scout", "geometry", (15,27), (18,27)))
        return got.env, next(iter(got.env.innovation_packets))
    def test_legal_features_ignore_hidden_truth(self):
        env, pid = self.source(); need = NeedPacket(0,"geometry",.25,1.,100.,0.,1)
        x = legal_need_features(env,"scout",pid,need); self.assertEqual(x.shape,(19,))
        other=env.clone(); other._truth.surface[:]=1-other._truth.surface
        np.testing.assert_array_equal(x,legal_need_features(other,"scout",pid,need))
    def test_deterministic_and_causally_guarded(self):
        env,pid=self.source(); actor=FrozenNeedA4()
        useful=NeedPacket(0,"geometry",.25,1.,100.,0.,1)
        self.assertEqual(actor.decide(env,"scout",pid,useful),actor.decide(env,"scout",pid,useful))
        wrong=NeedPacket(1,"geometry",.25,1.,100.,0.,2)
        self.assertFalse(actor.decide(env,"scout",pid,wrong)["send"])

if __name__ == "__main__": unittest.main()
