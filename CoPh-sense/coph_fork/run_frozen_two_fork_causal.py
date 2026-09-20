#!/usr/bin/env python3
"""Execute frozen A4/A5 through actual VMAS packets on the frozen map."""

import json

import numpy as np

from coph_fork.memory_critic_a5 import WORLDS
from coph_fork.oracle import ExactOracleConfig
from coph_fork.run_memory_a5 import load_models
from coph_fork.run_two_fork_memory_bridge import run_episode
from coph_fork.search_two_fork_physical_family import OUT
from coph_fork.two_fork_environment import TwoForkConfig, TwoForkWorld
from coph_fork.two_fork_memory import CarrierA5Adapter


def main():
    frozen = json.loads((OUT / "frozen_physical_layout_v1.json").read_text())
    layout = frozen["layout"]
    config = TwoForkConfig(
        horizon=frozen["horizon"],
        backup_y1=layout["backup_y1"],
        backup_y2=layout["backup_y2"],
        sensing_cost=layout["sensing_cost"])
    world = TwoForkWorld(backup_traction_1=layout["backup_traction_1"],
                         backup_traction_2=layout["backup_traction_2"])
    p1, p2 = layout["safe_prior_1"], layout["safe_prior_2"]
    prior = np.asarray([np.prod([
        probability if bit else 1 - probability
        for bit, probability in zip(bits, (p1, p1, p2, p2))])
        for bits in WORLDS], dtype=np.float64)
    _, a4 = load_models(1701, 1801)
    a5 = CarrierA5Adapter(config=ExactOracleConfig(
        sensing_cost=layout["sensing_cost"]))
    rows = []
    for condition in ("independent", "shared_reset", "persistent", "shuffled"):
        replay = OUT / f"frozen_{condition}_canonical_replay.json"
        if replay.exists():
            raise FileExistsError(replay)
        result = run_episode(world, condition, config=config, prior=prior,
                             a4_model=a4, a5_adapter=a5,
                             save_replay=replay)
        rows.append(result)
    output = OUT / "frozen_causal_canonical_v1.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(rows, indent=2) + "\n")
    for row in rows:
        print(row["condition"], row["chosen_acquisition"],
              round(row["team_cost"], 5), row["success"],
              row["ledger"]["collision"])


if __name__ == "__main__":
    main()
