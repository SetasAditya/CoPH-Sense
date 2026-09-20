import unittest

from coph_fork.memory_a5 import (
    EvidenceLedger, EvidenceRecord, exact_acquisition_value, oracle_carrier_action,
    route_for,
)
from coph_fork.oracle import ExactOracleConfig


def record(region, modality, value):
    return EvidenceRecord(
        evidence_id="r{}-{}".format(region, modality),
        region=region, modality=modality, value=value, uncertainty=0.0,
        acquisition_time=0, delivery_time=1, source="scout",
    )


class CooperativeMemoryA5Tests(unittest.TestCase):
    def test_persistent_memory_retains_value_and_provenance(self):
        memory = EvidenceLedger((record(1, "geometry", True),))
        copy = memory.copy()
        self.assertEqual(copy.value(1, "geometry"), True)
        self.assertEqual(copy.records[0].source, "scout")
        self.assertEqual(copy.records[0].delivery_time, 1)

    def test_known_pair_is_redundant_and_oracle_moves_to_region_two(self):
        config = ExactOracleConfig()
        memory = EvidenceLedger((
            record(1, "geometry", True), record(1, "traction", True)
        ))
        repeated = ((1, "geometry"), (1, "traction"))
        self.assertLess(exact_acquisition_value(memory, repeated, config), 0)
        self.assertEqual(
            oracle_carrier_action(memory, config),
            ((2, "geometry"), (2, "traction")),
        )

    def test_unknown_region_one_pair_has_positive_value(self):
        config = ExactOracleConfig()
        memory = EvidenceLedger()
        self.assertGreater(exact_acquisition_value(
            memory, ((1, "geometry"), (1, "traction")), config
        ), 0)

    def test_wrong_region_memory_does_not_resolve_original_region(self):
        memory = EvidenceLedger((
            record(1, "geometry", True), record(1, "traction", True)
        ))
        shuffled = memory.shuffled_region()
        self.assertEqual(route_for(memory, 1), "top")
        self.assertEqual(route_for(shuffled, 1), "bottom")
        self.assertEqual(route_for(shuffled, 2), "top")


if __name__ == "__main__":
    unittest.main()
