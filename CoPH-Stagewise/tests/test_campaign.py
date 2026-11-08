import json
import tempfile
import unittest
from pathlib import Path

from coph_stagewise import campaign


class CampaignTests(unittest.TestCase):
    def test_immutable_protocol_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first = campaign.freeze(root)
            self.assertEqual(campaign.freeze(root), first)
            self.assertTrue(campaign.verify(root)['protocol_hash_valid'])
            p = root / 'campaign_protocol.json'
            obj = json.loads(p.read_text())
            obj['protocol']['worker_limit'] = 99
            p.write_text(json.dumps(obj))
            with self.assertRaises(ValueError):
                campaign.verify(root)
            with self.assertRaises(ValueError):
                campaign.freeze(root)

    def test_missing_report_is_unavailable_not_success(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            campaign.freeze(root)
            rows = campaign.report(root)
            self.assertTrue(all(r['artifact_status'] == 'not_started' for r in rows))
            self.assertIn('Results unavailable', (root / 'REPORT.md').read_text())
            folder = root / 'reconstruction'
            folder.mkdir()
            (folder / 'REPORT.md').write_text('Gate failed; no treatments.')
            rows = campaign.inventory(root)
            row = next(r for r in rows if r['suite'] == 'reconstruction')
            self.assertEqual(row['artifact_status'], 'report_available')
            self.assertNotIn('passed', row)


if __name__ == '__main__':
    unittest.main()
