import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from coph_stagewise import campaign_reconstruction as c
from coph_stagewise.evaluation import load


class ReconstructionCampaignTests(unittest.TestCase):
    def test_symmetric_geometry_and_deterministic_fixtures(self):
        points = np.asarray(c.NODES)
        lengths = [sum(np.linalg.norm(points[i]-points[j]) for i,j in b) for b in c.BRANCHES]
        self.assertEqual(lengths[0], lengths[1])
        self.assertEqual(c.specification('upper_risk',82400), c.specification('upper_risk',82400))
        self.assertEqual(c.specification('upper_risk',82400)['patches'][0][1], .45)
        self.assertEqual(c.specification('lower_risk',82400)['patches'][0][1], -.45)
        self.assertEqual(c.specification('clear_control',82400)['patches'], [])

    def test_actual_branch_classification(self):
        self.assertIsNone(c.entered([1.5,0]))
        self.assertEqual(c.entered([2.,.3]), 'upper')
        self.assertEqual(c.entered([2.,-.3]), 'lower')
        self.assertIsNone(c.entered([0.,0.]))

    def test_six_arms_and_unchanged_budget(self):
        self.assertEqual(len(c.ARMS),6)
        self.assertEqual(set(c.ARMS.values()),{(fit,replay) for fit in (False,True) for replay in ('none','current','historical')})
        m=c.manifest()
        self.assertEqual(m['config']['replay_update_cap'],1024)
        self.assertEqual(m['seeds'],tuple(range(82400,82408)))
        self.assertFalse(m['held_out_executed'])

    def test_frozen_manifest_refuses_modified_configuration(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            with patch.object(c,'archive_sources'):
                original=c.freeze(root)
                self.assertEqual(c.freeze(root)['fingerprint'],original['fingerprint'])
                changed=copy.deepcopy(original)
                changed['fingerprint']='different'
                with patch.object(c,'manifest',return_value=changed):
                    with self.assertRaises(ValueError):c.freeze(root)
            self.assertEqual(load(root/'manifest.json')['fingerprint'],original['fingerprint'])


if __name__=='__main__':unittest.main()
