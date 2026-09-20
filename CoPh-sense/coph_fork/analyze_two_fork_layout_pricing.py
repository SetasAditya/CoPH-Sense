#!/usr/bin/env python3
"""Reweight exact paired VMAS continuations over public priors and sensing costs."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from coph_fork.search_two_fork_physical_family import Layout, summarize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--margin", type=float, default=.05)
    parser.add_argument("--local", action="store_true",
                        help="check a 3x3x3 neighborhood around (.65,.60,.02)")
    args = parser.parse_args()
    base = json.loads(args.result.read_text())
    base_layout = Layout(**base["layout"])
    reports = []
    priors_1 = (.60, .65, .70) if args.local else (.5, .65, .8)
    priors_2 = (.55, .60, .65) if args.local else (.5, .6, .7)
    prices = (.015, .02, .025) if args.local else (.01, .02, .04)
    for p1 in priors_1:
        for p2 in priors_2:
            for price in prices:
                layout = replace(base_layout, safe_prior_1=p1,
                                 safe_prior_2=p2, sensing_cost=price)
                worlds = []
                for original in base["worlds"]:
                    item = {**original,
                            "rows": [{**row, "downstream_cost":
                                      row["downstream_cost"] +
                                      (0 if row["action"] == "skip" else 2) *
                                      (price - base_layout.sensing_cost)}
                                     for row in original["rows"]]}
                    worlds.append(item)
                summary = summarize(worlds, layout, args.margin)
                reports.append({"safe_prior_1": p1, "safe_prior_2": p2,
                                "sensing_cost": price, "summary": summary})
    result = {"source": str(args.result), "scope":
              "exact repricing of fixed physical continuations; development search only",
              "results": reports}
    output = args.result.with_name(args.result.stem +
                                   ("_local_pricing.json" if args.local
                                    else "_pricing.json"))
    output.write_text(json.dumps(result, indent=2) + "\n")
    ranked = sorted(reports, key=lambda row: min(row["summary"]["contrast"].values()),
                    reverse=True)
    for row in ranked[:6]:
        print(row["safe_prior_1"], row["safe_prior_2"], row["sensing_cost"],
              "passed", row["summary"]["passed"],
              "min_margin", round(min(row["summary"]["contrast"].values()), 5),
              "contrast", {k: round(v, 5) for k, v in row["summary"]["contrast"].items()})
    print("passing", sum(row["summary"]["passed"] for row in reports),
          "of", len(reports), "worst_min_margin",
          round(min(min(row["summary"]["contrast"].values())
                    for row in reports), 5))


if __name__ == "__main__":
    main()
