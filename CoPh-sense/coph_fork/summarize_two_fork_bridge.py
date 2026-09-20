#!/usr/bin/env python3
"""Summarize frozen A5 choices against the exact VMAS acquisition reference."""

import json

import numpy as np

from coph_fork.memory_a5 import CANDIDATES, EvidenceLedger, EvidenceRecord
from coph_fork.run_memory_a5 import load_models
from coph_fork.run_two_fork_memory_bridge import OUT, protocol_posterior
from coph_fork.two_fork_environment import TwoForkConfig
from coph_fork.two_fork_memory import CarrierA5Adapter


def main():
    physical = json.loads((OUT / "physical_oracle_all_groups_symmetric_v2.json").read_text())
    independent = json.loads((OUT / "physical_oracle_all_groups_independent_probe_v2.json").read_text())
    canonical = json.loads((OUT / "canonical_symmetric_v2.json").read_text())
    _, a4 = load_models(1701, 1801)
    a5 = CarrierA5Adapter()
    rows = []
    for signature, reference in physical["by_memory_signature"].items():
        memory = EvidenceLedger(EvidenceRecord(
            evidence_id=f"received-{j}", region=region, modality=modality,
            value=value, uncertainty=0., acquisition_time=0,
            delivery_time=1, source="scout")
            for j, (region, modality, value) in enumerate(json.loads(signature)))
        posterior = protocol_posterior(memory, "persistent", a4, TwoForkConfig())
        index = CANDIDATES.index(a5.choose(memory, posterior))
        q = reference["q"]
        rows.append({"memory_signature": json.loads(signature),
                     "world_count": reference["world_count"],
                     "a5_action_index": index,
                     "a5_downstream_cost": q[index],
                     "oracle_action_index": reference["best_action_index"],
                     "oracle_downstream_cost": min(q),
                     "regret": q[index] - min(q),
                     "all_success": reference["all_success"]})
    no_memory = independent["by_memory_signature"]["[]"]
    report = {
        "scope": "symmetric moving two-fork restricted acquisition benchmark; frozen A4/A5",
        "persistent_by_signature": rows,
        "mean_persistent_regret": float(np.mean([r["regret"] for r in rows])),
        "mean_persistent_downstream_cost": float(np.mean([r["a5_downstream_cost"] for r in rows])),
        "mean_physical_oracle_downstream_cost": float(np.mean([r["oracle_downstream_cost"] for r in rows])),
        "no_memory_probe_q": no_memory["q"],
        "canonical": [{"condition": r["condition"],
                       "action": r["chosen_acquisition"],
                       "team_cost": r["team_cost"],
                       "success": r["success"]} for r in canonical],
    }
    path = OUT / "summary_symmetric_v2.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "mean_persistent_regret", "mean_persistent_downstream_cost",
        "mean_physical_oracle_downstream_cost", "no_memory_probe_q")}, indent=2))


if __name__ == "__main__":
    main()
