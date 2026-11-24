import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('baseline_audit', Path(__file__).with_name('audit.py'))
audit = importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)


class BaselineAdmissionTests(unittest.TestCase):
    def record(self):
        return dict(baseline='distgp', status='native_reproduced', adaptation_used_in_native_example=False,
                    source=dict(official_link_verified=True, repository_url='https://example.invalid/official',
                                commit='a'*40, license='declared-upstream-license'),
                    native_example=dict(command=['python','official_example.py'], input_hashes={'input':'0'*64},
                                        environment={'python':'test'}, log_sha256='0'*64,
                                        expected_checks=['finite_prediction'], return_code=0, checks_passed=True))

    def test_frozen_intent_integrity(self):
        self.assertFalse(audit.verify()['comparative_outcomes_observed'])

    def test_blocked_native_records_cannot_launch_comparison(self):
        records=json.loads((audit.ROOT/'native_reproduction.json').read_text())['baselines']
        for name,record in records.items():
            self.assertFalse(audit.admission(record)['eligible'])
            with tempfile.TemporaryDirectory() as d:
                native=Path(d)/'native.json'; native.write_text(json.dumps(record))
                out=Path(d)/'frozen.json'
                with self.assertRaisesRegex(ValueError,'Comparison gate blocked'):
                    audit.freeze(name,native,out)
                self.assertFalse(out.exists())

    def test_native_gate_requires_source_and_unadapted_passing_example(self):
        for key,value in [('commit',''),('official_link_verified',False),('license',None)]:
            r=self.record();r['source'][key]=value;self.assertFalse(audit.admission(r)['eligible'])
        for key,value in [('return_code',1),('checks_passed',False),('input_hashes',{})]:
            r=self.record();r['native_example'][key]=value;self.assertFalse(audit.admission(r)['eligible'])
        r=self.record();r['adaptation_used_in_native_example']=True;self.assertFalse(audit.admission(r)['eligible'])

    def test_freeze_is_idempotent_and_refuses_changed_record(self):
        with tempfile.TemporaryDirectory() as d:
            native=Path(d)/'native.json';out=Path(d)/'frozen.json';native.write_text(json.dumps(self.record()))
            first=audit.freeze('distgp',native,out)
            self.assertEqual(first,audit.freeze('distgp',native,out))
            r=self.record();r['source']['commit']='b'*40;native.write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'Refuse to replace'):
                audit.freeze('distgp',native,out)


if __name__ == '__main__': unittest.main()
