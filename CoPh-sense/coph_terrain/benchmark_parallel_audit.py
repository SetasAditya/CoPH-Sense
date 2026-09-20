"""End-to-end CPU parallelism benchmark for the restricted physical audit."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

from .physical_audit import audit_parent


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--parents", type=int, default=4)
    parser.add_argument("--seed-start", type=int, default=660010)
    parser.add_argument("--tag", default="parallel_cpu")
    args = parser.parse_args()
    if args.workers < 1 or args.parents < 1:
        raise ValueError("positive workers and parents are required")
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(audit_parent, ("open", "bottleneck", "labyrinth")[i % 3],
                               args.seed_start + i) for i in range(args.parents)]
        rows = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    continuation_count = sum(len(row["results"]) for row in rows)
    output = {"workers": args.workers, "parents": args.parents,
              "continuations": continuation_count,
              "wall_seconds": elapsed,
              "complete_continuations_per_second": continuation_count / elapsed,
              "parent_wall_seconds": [row["wall_seconds"] for row in rows],
              "parent_seeds": [row["parent_seed"] for row in rows]}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"physical_audit_throughput_{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    print(path)


if __name__ == "__main__":
    main()
