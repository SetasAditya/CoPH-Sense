import json
from pathlib import Path
import tempfile
import unittest

from coph_terrain.confirmatory_campaign import (
    METHODS, aggregate, load_records, load_schedule, run_campaign, write_report)
from coph_terrain.confirmatory_smoke_adapter import evaluate


class ConfirmatoryCampaignTests(unittest.TestCase):
    def test_paired_resumable_campaign_and_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root/"test.json"
            manifest.write_text(json.dumps({"split": "test", "groups": [
                {"group_id": "test/open/0000", "family": "open",
                 "parent_seed": 41}], "episode_realizations_per_test_group": 2}))
            keys = load_schedule([manifest], [7], realizations=2)
            output = root/"episodes.jsonl"
            self.assertEqual(run_campaign(keys, evaluate, output), 2*len(METHODS))
            self.assertEqual(run_campaign(keys, evaluate, output), 2*len(METHODS))
            rows = load_records(output)
            report = aggregate(rows)
            self.assertEqual(report["paired_blocks"], 2)
            self.assertEqual(set(report["methods"]["test"]), set(METHODS))
            write_report(report, root, "smoke")
            self.assertTrue((root/"smoke_table.md").exists())

    def test_rejects_incomplete_pair(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"bad.jsonl"
            path.write_text(json.dumps({"schema_version": 1, "record_id": "x",
                "method": METHODS[0], "split": "test", "training_seed": 0,
                "group_id": "g", "realization": 0})+"\n")
            with self.assertRaises(ValueError):
                load_records(path)


if __name__ == "__main__":
    unittest.main()
