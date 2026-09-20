#!/usr/bin/env python3
"""Predeclared useful-shortcut stratum for balanced VMAS gate evaluation."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp

import numpy as np

from coph_fork.vmas_gate_data_a65 import (
    DATA, GateContext, init_worker, label_context,
)


def contexts(split, count, seed):
    rng = np.random.RandomState(seed)
    result = []
    for i in range(count):
        held_out = split == "ood"
        result.append(GateContext(
            key=f"{split}-useful-{i:03d}", split=split,
            bottom_traction_mean=float(rng.uniform(.14, .20)
                                       if held_out else rng.uniform(.15, .24)),
            sensing_cost=float(rng.uniform(.015, .065)
                               if held_out else rng.uniform(.025, .10)),
            delay_steps=int(rng.choice((4, 15)) if held_out
                            else rng.choice((2, 12, 25))),
            delivery_probability=float(rng.choice((.85, 1.0)) if held_out
                                       else rng.choice((.80, 1.0))),
            communication_range=float(rng.uniform(.48, .63) if held_out
                                      else rng.uniform(.70, .95)),
            safe_top_prior=float(rng.uniform(.53, .68) if held_out
                                 else rng.uniform(.38, .60)),
            carrier_start_y=float(rng.uniform(.03, .065)),
            scout_start_y=float(rng.uniform(.38, .42)),
            bottom_route_y=float(rng.uniform(-.66, -.63)),
            carrier_top_y=.32,
        ))
    return result


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    targets = contexts("train", 8, 701) + contexts("val", 2, 702) + contexts("ood", 3, 703)
    (DATA / "augmentation_contexts.json").write_text(
        json.dumps([asdict(context) for context in targets], indent=2) + "\n")
    output = DATA / "labels.json"
    rows = json.loads(output.read_text())
    done = {row["context"]["key"] for row in rows}
    with ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context("spawn"),
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
