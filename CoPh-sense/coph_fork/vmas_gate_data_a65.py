#!/usr/bin/env python3
"""Paired physical acquisition-advantage labels for A6.5.

Each label is an ex-ante VMAS value under a public four-world top-terrain
prior. The scout gate receives public task metadata and current local/public
kinematics, never the sampled top geometry or traction. Packet outcomes are
enumerated exactly under the fixed two-message A4 continuation.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import torch

from coph_fork.environment import ForkWorld
from coph_fork.run_moving_bridge_a65 import (
    OUT, gate_snapshot, load_actors, run_episode, state, timing_context,
)
from coph_fork.run_moving_oracle_a65_v3 import BASE


DATA = OUT / "gate_data"
_A4 = None
_FINITE_GATE = None


@dataclass(frozen=True)
class GateContext:
    key: str
    split: str
    bottom_traction_mean: float
    sensing_cost: float
    delay_steps: int
    delivery_probability: float
    communication_range: float
    safe_top_prior: float
    carrier_start_y: float
    scout_start_y: float
    bottom_route_y: float
    carrier_top_y: float

    def config(self):
        return replace(
            BASE,
            backup_traction_prior_mean=self.bottom_traction_mean,
            sensing_cost=self.sensing_cost,
            communication_delay_steps=self.delay_steps,
            packet_drop_probability=1 - self.delivery_probability,
            communication_range=self.communication_range,
            carrier_start_y=self.carrier_start_y,
            scout_start_y=self.scout_start_y,
            bottom_route_y=self.bottom_route_y,
            carrier_top_y=self.carrier_top_y,
        )


def public_features(context, decision_state):
    """Pre-acquisition input; no ForkWorld values or post-decision outcomes."""
    config = decision_state.config
    scout_pos, scout_vel = state(decision_state, "scout")
    carrier_pos, carrier_vel = state(decision_state, "carrier")
    timing = timing_context(decision_state, rollout_forecast=True)
    values = [
        *scout_pos, *scout_vel, *carrier_pos, *carrier_vel,
        config.backup_traction_prior_mean, context.safe_top_prior,
        config.carrier_top_y, config.bottom_route_y, config.goal_y,
        config.sensing_cost, config.communication_range,
        context.delivery_probability, config.communication_delay_steps * config.dt,
        timing.decision_slack, timing.return_distance,
        timing.delivery_time, timing.temporal_margin,
    ]
    return [float(value) for value in values]


def world_weight(geometry, traction, safe_prior):
    return safe_prior if geometry and traction == 0.90 else (1 - safe_prior) / 3


def drop_masks(probability):
    for mask in product((False, True), repeat=2):
        weight = np.prod([1 - probability if dropped else probability
                          for dropped in mask])
        if weight > 0:
            yield mask, float(weight)


def init_worker():
    global _A4, _FINITE_GATE
    torch.set_num_threads(1)
    _A4, _FINITE_GATE = load_actors()


def label_context(context):
    config = context.config()
    world_rows = []
    reference_features = None
    reference_observation = None
    for geometry, traction in product((False, True), (0.30, 0.90)):
        world = ForkWorld(top_geometry=geometry, top_traction=traction,
                          bottom_geometry=True,
                          bottom_traction=context.bottom_traction_mean)
        snapshot = gate_snapshot(world, config, 1701)
        observation = snapshot.observations()["scout"]
        candidate_features = public_features(context, snapshot)
        if reference_observation is None:
            reference_observation = observation
            reference_features = candidate_features
        elif observation != reference_observation:
            raise AssertionError("hidden top world leaked into the gate observation")
        elif not np.allclose(candidate_features, reference_features, atol=1e-6):
            raise AssertionError("hidden top world changed the gate forecast")
        skip = run_episode(world, config, "never", 1701, _A4, _FINITE_GATE,
                           starting_env=snapshot, rollout_forecast=True)
        skip_cost = skip["team_cost"] - snapshot.ledger.total
        inspect_cost = 0.0
        for mask, weight in drop_masks(context.delivery_probability):
            result = run_episode(world, config, "always", 1701,
                                 _A4, _FINITE_GATE,
                                 starting_env=snapshot, rollout_forecast=True,
                                 forced_evidence_drops=mask)
            inspect_cost += weight * (result["team_cost"] - snapshot.ledger.total)
        world_rows.append({"geometry": geometry, "traction": traction,
                           "weight": world_weight(geometry, traction,
                                                  context.safe_top_prior),
                           "inspect": inspect_cost, "skip": skip_cost})
    inspect = sum(row["weight"] * row["inspect"] for row in world_rows)
    skip = sum(row["weight"] * row["skip"] for row in world_rows)
    return {"context": asdict(context), "features": reference_features,
            "q_inspect": inspect, "q_skip": skip,
            "advantage": skip - inspect,
            "world_values": world_rows}


def rough_advantage(context):
    """Cheap proposal filter only; labels always come from VMAS rollouts."""
    backup = 0.61 + 0.36 / context.bottom_traction_mean
    probe = 2 * context.sensing_cost + 0.049
    timely = context.delay_steps * BASE.dt + 0.1 < 4.2
    delivery = context.delivery_probability ** 2
    return (context.safe_top_prior * delivery * (backup - 0.93) - probe
            if timely else -probe)


def sample_contexts(count, split, seed, near_fraction=0.45):
    rng = np.random.RandomState(seed)
    chosen = []
    attempts = 0
    while len(chosen) < count and attempts < count * 300:
        attempts += 1
        bottom = float(rng.uniform(0.15, 0.55)
                       if split != "ood" else rng.choice((0.14, 0.58)))
        sensing = float(rng.uniform(0.025, 0.32)
                        if split != "ood" else rng.choice((0.015, 0.40)))
        p = float(rng.choice((1.0, 0.65, 0.25), p=(0.68, 0.22, 0.10))
                  if split != "ood" else rng.choice((0.40, 0.85)))
        delay = int(rng.choice((2, 12, 32, 65, 110), p=(.40, .15, .20, .15, .10))
                    if split != "ood" else rng.choice((4, 48, 85)))
        context = GateContext(
            key=f"{split}-{len(chosen):03d}", split=split,
            bottom_traction_mean=bottom, sensing_cost=sensing,
            delay_steps=delay, delivery_probability=p,
            communication_range=float(rng.uniform(0.55, 0.95)
                                      if split != "ood" else rng.uniform(0.43, 0.55)),
            safe_top_prior=float(rng.uniform(0.15, 0.55)
                                 if split != "ood" else rng.choice((0.10, 0.65))),
            carrier_start_y=float(rng.uniform(0.03, 0.065)),
            scout_start_y=float(rng.uniform(0.38, 0.42)),
            bottom_route_y=float(rng.uniform(-0.66, -0.63)),
            carrier_top_y=BASE.carrier_top_y,
        )
        if len(chosen) < int(count * near_fraction) and abs(rough_advantage(context)) > .10:
            continue
        chosen.append(context)
    if len(chosen) != count:
        raise RuntimeError("could not sample enough proposed contexts")
    return chosen


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=20)
    parser.add_argument("--val", type=int, default=6)
    parser.add_argument("--ood", type=int, default=6)
    parser.add_argument("--jobs", type=int, default=6)
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    contexts = (sample_contexts(args.train, "train", 601)
                + sample_contexts(args.val, "val", 602)
                + sample_contexts(args.ood, "ood", 603, near_fraction=0))
    (DATA / "contexts.json").write_text(json.dumps([asdict(c) for c in contexts], indent=2) + "\n")
    output = DATA / "labels.json"
    rows = json.loads(output.read_text()) if output.exists() else []
    completed = {row["context"]["key"] for row in rows}
    todo = [context for context in contexts if context.key not in completed]
    with ProcessPoolExecutor(max_workers=args.jobs,
                             mp_context=mp.get_context("spawn"),
                             initializer=init_worker) as pool:
        futures = {pool.submit(label_context, context): context for context in todo}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2) + "\n")
            print(row["context"]["key"], round(row["advantage"], 4), flush=True)
    print("labeled", len(rows), "contexts")


if __name__ == "__main__":
    main()
