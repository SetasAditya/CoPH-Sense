#!/usr/bin/env python3
"""First frozen A5-v2 evaluation on the oracle-selected moving layout."""

import hashlib
import json

import numpy as np

from coph_fork.memory_a5 import EvidenceLedger, EvidenceRecord
from coph_fork.memory_critic_a5 import WORLDS, posterior_for
from coph_fork.oracle import ExactOracleConfig
from coph_fork.search_two_fork_physical_family import OUT
from coph_fork.two_fork_memory import CarrierA5Adapter


NAME_BY_ACTION = {
    (): "skip",
    ((1, "geometry"),): "R1_G",
    ((1, "traction"),): "R1_T",
    ((1, "geometry"), (1, "traction")): "R1",
    ((2, "geometry"),): "R2_G",
    ((2, "traction"),): "R2_T",
    ((2, "geometry"), (2, "traction")): "R2",
}


def main():
    frozen_path = OUT / "frozen_physical_layout_v1.json"
    frozen = json.loads(frozen_path.read_text())
    here = OUT.parents[2]
    for relative, expected in frozen["sha256"].items():
        actual = hashlib.sha256((here / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"frozen physical file changed: {relative}")
    layout = frozen["layout"]
    p1, p2 = layout["safe_prior_1"], layout["safe_prior_2"]
    prior = []
    for bits in WORLDS:
        mass = 1.
        for bit, probability in zip(bits, (p1, p1, p2, p2)):
            mass *= probability if bit else 1 - probability
        prior.append(mass)
    prior = np.asarray(prior, dtype=np.float64)
    memory_states = {
        "M0": EvidenceLedger(),
        "M1": EvidenceLedger([
            EvidenceRecord(f"scout-r1-{modality}", 1, modality, True,
                           0., 0, 1, "scout")
            for modality in ("geometry", "traction")]),
    }
    model = CarrierA5Adapter(config=ExactOracleConfig(
        sensing_cost=layout["sensing_cost"]))
    rows = []
    for state, memory in memory_states.items():
        posterior = posterior_for(memory, prior)
        action = model.choose(memory, posterior)
        name = NAME_BY_ACTION[action]
        q = frozen["q"][state]
        rows.append({"memory_state": state, "a5_action": name,
                     "physical_oracle_action": frozen["best"][state],
                     "a5_physical_cost": q[name],
                     "oracle_physical_cost": min(q.values()),
                     "physical_regret": q[name] - min(q.values()),
                     "posterior": posterior.tolist()})
    paired_worlds = json.loads((OUT / "layout_01_geometry.json").read_text())["worlds"]
    matched = {}
    for memory_state, action in (("M0", "R1"), ("M1", "R2")):
        cost = weight = 0.
        for item in paired_worlds:
            bits = item["bits"]
            if bits[:2] != [True, True]:
                continue
            mass = ((p2 if bits[2] else 1 - p2) *
                    (p2 if bits[3] else 1 - p2))
            row = next(r for r in item["rows"]
                       if r["memory"] == memory_state and r["action"] == action)
            cost += mass * row["downstream_cost"]
            weight += mass
        matched[memory_state] = cost / weight
    result = {"scope": "frozen finite A5-v2 actor; first evaluation on frozen matched moving VMAS layout",
              "checkpoint": "results/a5_memory_v2/pairwise_1701.pth",
              "frozen_layout_sha256": hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
              "rows": rows,
              "matched_r1_safe_worlds": {
                  "no_memory_remeasure_r1_cost": matched["M0"],
                  "valid_memory_sense_r2_cost": matched["M1"],
                  "cost_reduction": matched["M0"] - matched["M1"],
                  "same_staging_prefix": True,
              },
              "both_decisions_match": all(row["a5_action"] == row["physical_oracle_action"]
                                          for row in rows)}
    path = OUT / "frozen_a5_evaluation_v2.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"both_decisions_match": result["both_decisions_match"],
                      "matched_r1_safe_worlds": result["matched_r1_safe_worlds"],
                      "rows": [{key: value for key, value in row.items()
                                if key != "posterior"} for row in rows]}, indent=2))


if __name__ == "__main__":
    main()
