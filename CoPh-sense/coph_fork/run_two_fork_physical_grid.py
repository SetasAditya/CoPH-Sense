#!/usr/bin/env python3
"""Evaluate a predeclared oracle-only VMAS geometry grid without A5 access."""

import argparse
import json
from pathlib import Path

from coph_fork.search_two_fork_physical_family import (
    Layout, OUT, evaluate_layout,
)


HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=Path,
                        default=HERE / "physical_search_grid_v1.json")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    spec = json.loads(args.grid.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for item in spec["layouts"]:
        name = item["name"]
        layout = Layout(**{key: value for key, value in item.items()
                           if key != "name"})
        path = OUT / f"grid_v1_{name}.json"
        if path.exists():
            result = json.loads(path.read_text())
            if result["layout"] != vars(layout):
                raise ValueError(f"existing {path} has different parameters")
        else:
            result = evaluate_layout(layout, args.jobs, spec["margin"])
            path.write_text(json.dumps(result, indent=2) + "\n")
        summaries.append({"name": name, "path": str(path),
                          "layout": result["layout"],
                          "summary": result["summary"]})
        print(name, result["summary"]["contrast"],
              "pass", result["summary"]["passed"], flush=True)
    report = {"grid": str(args.grid), "scope": spec["scope"],
              "margin": spec["margin"], "candidates": summaries,
              "accepted": [row["name"] for row in summaries
                           if row["summary"]["passed"]]}
    (OUT / "grid_v1_summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
