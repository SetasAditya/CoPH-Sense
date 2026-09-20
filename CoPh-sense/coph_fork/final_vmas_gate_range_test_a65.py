#!/usr/bin/env python3
"""Second untouched A6.5 test, emphasizing radio-range compositions."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import multiprocessing as mp

import numpy as np

from coph_fork.vmas_gate_data_a65 import DATA, GateContext, init_worker, label_context


def plan():
    rng = np.random.RandomState(1001)
    result = []
    for stratum, count in (("open_link", 3), ("range_challenge", 5),
                           ("cost_delay", 4)):
        for i in range(count):
            if stratum == "open_link":
                bottom, sensing = rng.uniform(.15, .22), rng.uniform(.03, .10)
                delay, p = int(rng.randint(2, 24)), 1.0
                prior, radio = rng.uniform(.42, .65), rng.uniform(.76, .94)
            elif stratum == "range_challenge":
                bottom, sensing = rng.uniform(.15, .24), rng.uniform(.03, .11)
                delay, p = int(rng.randint(2, 45)), 1.0
                prior, radio = rng.uniform(.38, .65), rng.uniform(.44, .64)
            else:
                bottom, sensing = rng.uniform(.14, .41), rng.uniform(.18, .40)
                delay, p = int(rng.randint(20, 105)), float(rng.choice((.40, .80)))
                prior, radio = rng.uniform(.20, .60), rng.uniform(.48, .82)
            result.append(GateContext(
                key=f"range-final-{stratum}-{i:02d}", split="range_final",
                bottom_traction_mean=float(bottom), sensing_cost=float(sensing),
                delay_steps=delay, delivery_probability=float(p),
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
    (DATA / "range_final_contexts.json").write_text(
        json.dumps([asdict(context) for context in targets], indent=2) + "\n")
    if args.plan_only:
        print("planned", len(targets), "untouched range contexts")
        return
    output = DATA / "range_final_labels.json"
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
