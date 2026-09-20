#!/usr/bin/env python3
"""One final frozen test: paired actionability deadlines plus cost boundaries."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import multiprocessing as mp

import numpy as np

from coph_fork.environment import ForkWorld
from coph_fork.run_moving_bridge_a65 import gate_snapshot
from coph_fork.vmas_gate_data_a65 import DATA, GateContext, init_worker, label_context
from coph_fork.vmas_gate_trajectory_a65 import actionability_features, reconnect_features


def make_context(regime, index, rng):
    bottom = float(rng.uniform(.16, .30))
    prior = float(rng.uniform(.3, .6))
    probability = float(rng.uniform(.80, 1.0))
    radio = float(rng.uniform(.78, .94))
    if regime == "range":
        radio = float(rng.uniform(.44, .62))
    if regime == "packet":
        probability = float(rng.uniform(.3, .7))
    return GateContext(
        key=f"action-anchor-{regime}-{index}", split="action_anchor",
        bottom_traction_mean=bottom, sensing_cost=.08, delay_steps=3,
        delivery_probability=probability, communication_range=radio,
        safe_top_prior=prior,
        carrier_start_y=float(rng.uniform(.03, .065)),
        scout_start_y=float(rng.uniform(.38, .42)),
        bottom_route_y=float(rng.uniform(-.66, -.63)), carrier_top_y=.32,
    )


def main():
    rng = np.random.RandomState(1207)
    contexts_file = DATA / "final_actionability_contexts.json"
    labels_file = DATA / "final_actionability_labels.json"
    if contexts_file.exists() or labels_file.exists():
        raise FileExistsError("final actionability set is frozen")
    deadline = []
    for index in range(5):
        base = make_context("deadline", index, rng)
        world = ForkWorld(top_geometry=True, top_traction=.9,
                          bottom_geometry=True,
                          bottom_traction=base.bottom_traction_mean)
        state = gate_snapshot(world, base.config(), 1701)
        reconnect = reconnect_features(state)
        latest = actionability_features(state, reconnect)[0]
        center = round((latest - 2 * base.config().dt - reconnect[0]) / base.config().dt)
        for side, offset in (("early", -2), ("late", 2)):
            deadline.append(replace(base,
                key=f"action-final-deadline-{index}-{side}",
                split="final_actionability", sensing_cost=.04,
                delay_steps=max(0, center + offset)))
    anchors = [make_context(regime, i, rng)
               for regime, count in (("range", 10), ("probe", 7), ("packet", 7))
               for i in range(count)]
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                             initializer=init_worker) as pool:
        base_labels = list(pool.map(label_context, anchors))
    adjusted = []
    selected_by_regime = {name: 0 for name in ("range", "probe", "packet")}
    for context, row in zip(anchors, base_labels):
        regime = next(name for name in selected_by_regime if name in context.key)
        if selected_by_regime[regime] >= 5:
            continue
        center = context.sensing_cost + row["advantage"] / 2
        if not .02 <= center <= .4:
            continue
        side = 1 if len(adjusted) % 2 else -1
        adjusted.append(replace(context,
            key=context.key.replace("anchor", "final"),
            split="final_actionability",
            sensing_cost=float(center + side * .0125)))
        selected_by_regime[regime] += 1
    counts = {regime: sum(regime in c.key for c in adjusted)
              for regime in ("range", "probe", "packet")}
    if any(counts[r] < 5 for r in counts):
        raise RuntimeError(f"too few feasible cost-boundary contexts: {counts}")
    final = deadline + adjusted
    contexts_file.write_text(json.dumps([asdict(c) for c in final], indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                             initializer=init_worker) as pool:
        labels = list(pool.map(label_context, final))
    labels_file.write_text(json.dumps(labels, indent=2) + "\n")
    print({"count": len(labels), "strata": {r: sum(r in c.key for c in final)
                                         for r in ("deadline", "range", "probe", "packet")},
           "positive": sum(r["advantage"] > 0 for r in labels)})


if __name__ == "__main__":
    main()
