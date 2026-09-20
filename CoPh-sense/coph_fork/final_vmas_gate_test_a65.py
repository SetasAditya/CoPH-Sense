#!/usr/bin/env python3
"""Predeclared untouched combination test for the A6.5 VMAS gate."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp

import numpy as np

from coph_fork.vmas_gate_data_a65 import DATA, GateContext, init_worker, label_context


def plan():
    rng = np.random.RandomState(901)
    result = []
    for stratum in ("candidate_positive", "candidate_negative"):
        for i in range(6):
            if stratum == "candidate_positive":
                bottom = rng.uniform(.15, .23)
                sensing = rng.uniform(.025, .11)
                delay = int(rng.choice((8, 40)))
                p = float(rng.choice((.75, 1.0)))
                prior = rng.uniform(.43, .65)
                radio = rng.uniform(.63, .93)
            else:
                bottom = rng.uniform(.30, .56)
                sensing = rng.uniform(.14, .38)
                delay = int(rng.choice((45, 82, 125)))
                p = float(rng.choice((.45, .75)))
                prior = rng.uniform(.17, .48)
                radio = rng.uniform(.50, .85)
            result.append(GateContext(
                key=f"final-{stratum}-{i:02d}", split="final",
                bottom_traction_mean=float(bottom), sensing_cost=float(sensing),
                delay_steps=delay, delivery_probability=p,
                communication_range=float(radio), safe_top_prior=float(prior),
                carrier_start_y=float(rng.uniform(.035, .06)),
                scout_start_y=float(rng.uniform(.38, .42)),
                bottom_route_y=float(rng.uniform(-.66, -.63)),
                carrier_top_y=.32,
            ))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    targets = plan()
    (DATA / "final_contexts.json").write_text(
        json.dumps([asdict(context) for context in targets], indent=2) + "\n")
    if args.plan_only:
        print("planned", len(targets), "untouched final contexts")
        return
    output = DATA / "final_labels.json"
    rows = json.loads(output.read_text()) if output.exists() else []
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
