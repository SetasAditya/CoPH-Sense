#!/usr/bin/env python3
"""Freeze a physical near-boundary test before residual-gate training."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import multiprocessing as mp

import numpy as np

from coph_fork.vmas_gate_data_a65 import DATA, GateContext, init_worker, label_context


def candidates():
    rng = np.random.RandomState(1107)
    result = []
    for regime in ("range", "probe", "deadline", "packet"):
        for index in range(8):
            bottom = float(rng.uniform(.15, .30))
            prior = float(rng.uniform(.32, .65))
            probability = float(rng.uniform(.8, 1.0))
            delay = int(rng.randint(2, 22))
            radio = float(rng.uniform(.72, .93))
            if regime == "range":
                radio = float(rng.uniform(.42, .62))
            elif regime == "deadline":
                delay = int(rng.randint(65, 110))
            elif regime == "packet":
                probability = float(rng.uniform(.35, .75))
            result.append(GateContext(
                key=f"boundary-anchor-{regime}-{index}", split="boundary_anchor",
                bottom_traction_mean=bottom, sensing_cost=.08,
                delay_steps=delay, delivery_probability=probability,
                communication_range=radio, safe_top_prior=prior,
                carrier_start_y=float(rng.uniform(.03, .065)),
                scout_start_y=float(rng.uniform(.38, .42)),
                bottom_route_y=float(rng.uniform(-.66, -.63)), carrier_top_y=.32,
            ))
    return result


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "untouched_boundary_contexts.json"
    labels_path = DATA / "untouched_boundary_labels.json"
    if path.exists() or labels_path.exists():
        raise FileExistsError("boundary set is frozen; refusing to regenerate")
    anchors = candidates()
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                             initializer=init_worker) as pool:
        base = list(pool.map(label_context, anchors))
    adjusted = []
    for anchor, row in zip(anchors, base):
        # For this fixed continuation, changing a probe's charge changes
        # inspect advantage by exactly -2 Delta cost. Alternate either side.
        center = anchor.sensing_cost + row["advantage"] / 2
        if not .025 <= center <= .40:
            continue
        side = -1 if len(adjusted) % 2 else 1
        cost = center + side * .0125  # expected absolute advantage ~.025
        if not .015 <= cost <= .42:
            continue
        adjusted.append(replace(
            anchor, key=anchor.key.replace("anchor", "final"),
            split="untouched_boundary", sensing_cost=float(cost)))
    if len(adjusted) < 12:
        raise RuntimeError(f"only {len(adjusted)} feasible boundary contexts")
    # Freeze contexts before producing final labels or training a new model.
    path.write_text(json.dumps([asdict(c) for c in adjusted], indent=2) + "\n")
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                             initializer=init_worker) as pool:
        rows = list(pool.map(label_context, adjusted))
    if any(abs(row["advantage"]) >= .05 for row in rows):
        raise AssertionError("a frozen context is not near the physical boundary")
    labels_path.write_text(json.dumps(rows, indent=2) + "\n")
    print({"count": len(rows),
           "by_regime": {name: sum(name in r["context"]["key"] for r in rows)
                         for name in ("range", "probe", "deadline", "packet")},
           "positive": sum(r["advantage"] > 0 for r in rows),
           "max_abs_advantage": max(abs(r["advantage"]) for r in rows)})


if __name__ == "__main__":
    main()
