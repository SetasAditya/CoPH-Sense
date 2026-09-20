"""Blind pretraining audit of topology/risk shortcuts on excluded parent maps."""

import argparse
import json
from pathlib import Path

import numpy as np

from .generator import FAMILIES, GOAL_CELL, START_CELL, generate_map, realize_map
from .planning import dijkstra


OUT = Path(__file__).resolve().parent / "results" / "pilot"
RISK_WEIGHTS = (0., .5, 1., 2., 3., 5.)


def _path(cost, start=START_CELL, goal=GOAL_CELL):
    distances, previous = dijkstra(cost, start)
    if not np.isfinite(distances[goal]):
        return ()
    path = [goal]
    while path[-1] != start:
        preceding = int(previous[path[-1]])
        path.append((preceding // 60, preceding % 60))
    return tuple(reversed(path))


def summarize_map(family, seed):
    terrain = realize_map(generate_map(family, seed), 0)
    blocked = terrain.occupancy.copy()
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            blocked[max(0, dx):60 + min(0, dx),
                    max(0, dy):60 + min(0, dy)] |= terrain.occupancy[
                        max(0, -dx):60 - max(0, dx),
                        max(0, -dy):60 - max(0, dy)]
    geometric_cost = np.where(blocked, np.inf, 1.)
    geometric = _path(geometric_cost)
    routes = [_path(np.where(blocked, np.inf, 1. + weight * terrain.risk))
              for weight in RISK_WEIGHTS]
    safer = routes[-2]
    if not geometric or not safer:
        raise RuntimeError(f"reachable generator map lost its route: {family}/{seed}")
    reachable, _ = dijkstra(geometric_cost, START_CELL)
    reachable_cells = np.isfinite(reachable)
    geometric_set, safer_set = set(geometric), set(safer)
    distinct_fraction = 1. - len(geometric_set & safer_set) / len(geometric_set | safer_set)
    route_classes = []
    for path in routes:
        cells = set(path)
        if all(1. - len(cells & old) / len(cells | old) > .20
               for old in route_classes):
            route_classes.append(cells)
    return {
        "family": family,
        "seed": int(seed),
        "risk_coverage_over_free": float(np.mean(terrain.risk[~terrain.occupancy] > .5)),
        "risk_coverage_over_reachable": float(np.mean(terrain.risk[reachable_cells] > .5)),
        "mean_risk_geometric_shortest": float(np.mean([terrain.risk[cell] for cell in geometric])),
        "geometric_path_cells": len(geometric),
        "risk_aware_path_cells": len(safer),
        "route_jaccard_distance": float(distinct_fraction),
        "materially_distinct_route": bool(distinct_fraction > .20),
        "sampled_route_classes": len(route_classes),
        "risk_path_length_inflation": float(len(safer) / len(geometric) - 1.),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-family", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=650000)
    args = parser.parse_args()
    if args.per_family < 1:
        raise ValueError("per-family must be positive")
    rows = [summarize_map(family, args.seed_start + i * args.per_family + j)
            for i, family in enumerate(FAMILIES)
            for j in range(args.per_family)]
    metrics = ("risk_coverage_over_free", "risk_coverage_over_reachable",
               "mean_risk_geometric_shortest", "route_jaccard_distance",
               "materially_distinct_route", "sampled_route_classes",
               "risk_path_length_inflation")
    summary = {family: {metric: {
                    "mean": float(np.mean([row[metric] for row in rows
                                           if row["family"] == family])),
                    "p10_p50_p90": np.quantile(
                        [float(row[metric]) for row in rows
                         if row["family"] == family],
                        [.1, .5, .9]).tolist(),
                }
                        for metric in metrics}
               for family in FAMILIES}
    payload = {"purpose": "pretraining generator audit only; excludes all E2 split seeds",
               "per_family": args.per_family, "seed_start": args.seed_start,
               "summary": summary, "maps": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"generator_audit_v2_{args.per_family}_per_family.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
