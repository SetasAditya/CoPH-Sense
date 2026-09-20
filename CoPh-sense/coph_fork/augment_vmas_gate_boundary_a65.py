#!/usr/bin/env python3
"""One focused development augmentation for weak-link, late, and low-value cases."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp

import numpy as np

from coph_fork.vmas_gate_data_a65 import DATA, GateContext, init_worker, label_context


def contexts(split, per_regime, seed):
    rng = np.random.RandomState(seed)
    result = []
    for regime in ("low_value", "late", "weak_link"):
        for i in range(per_regime):
            if regime == "low_value":
                bottom = rng.uniform(.40, .58)
                sensing = rng.uniform(.025, .12)
                delay = int(rng.randint(2, 25))
                p = rng.uniform(.80, 1.0)
                prior = rng.uniform(.20, .45)
            elif regime == "late":
                bottom = rng.uniform(.14, .25)
                sensing = rng.uniform(.025, .12)
                delay = int(rng.randint(105, 176))
                p = 1.0
                prior = rng.uniform(.30, .60)
            else:
                bottom = rng.uniform(.14, .25)
                sensing = rng.uniform(.025, .09)
                delay = int(rng.randint(2, 24))
                p = rng.uniform(.10, .35)
                prior = rng.uniform(.30, .60)
            result.append(GateContext(
                key=f"{split}-boundary-{regime}-{i:02d}", split=split,
                bottom_traction_mean=float(bottom), sensing_cost=float(sensing),
                delay_steps=delay, delivery_probability=float(p),
                communication_range=float(rng.uniform(.68, .95)),
                safe_top_prior=float(prior),
                carrier_start_y=float(rng.uniform(.03, .065)),
                scout_start_y=float(rng.uniform(.38, .42)),
                bottom_route_y=float(rng.uniform(-.66, -.63)),
                carrier_top_y=.32,
            ))
    return result


def main():
    targets = contexts("train", 6, 801) + contexts("val", 1, 802)
    (DATA / "boundary_augmentation_contexts.json").write_text(
        json.dumps([asdict(context) for context in targets], indent=2) + "\n")
    output = DATA / "labels.json"
    rows = json.loads(output.read_text())
    done = {row["context"]["key"] for row in rows}
    with ProcessPoolExecutor(max_workers=8, mp_context=mp.get_context("spawn"),
                             initializer=init_worker) as pool:
        futures = [pool.submit(label_context, context) for context in targets
                   if context.key not in done]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            output.write_text(json.dumps(rows, indent=2) + "\n")
            print(row["context"]["key"], round(row["advantage"], 4), flush=True)


if __name__ == "__main__":
    main()
