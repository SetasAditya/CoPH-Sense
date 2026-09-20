"""Post hoc diagnostic of SEND value under identical deployed feature vectors.

The within-split group oracle uses true realized values, so it is descriptive
and cannot be deployed or used for checkpoint selection. Exact equality is
used deliberately; this makes the empirical statement fully reproducible.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


FEATURES = ("actor_features", "recipient_features", "response_features")


def audit(rows, feature):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[feature])].append(row["decision_value"])
    positive_mean = [values for values in groups.values()
                     if sum(values) > 0]
    mixed = [values for values in groups.values()
             if any(value > 0 for value in values) and
             any(value <= 0 for value in values)]
    return {"rows": len(rows), "realized_positive_send": sum(
                row["decision_value"] > 0 for row in rows),
            "distinct_exact_inputs": len(groups),
            "mixed_sign_inputs": len(mixed),
            "positive_mean_inputs": len(positive_mean),
            "within_split_hindsight_group_gain_over_never": (
                sum(max(0., sum(values)) for values in groups.values()) /
                len(rows))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path)
    args = parser.parse_args()
    directory = args.dataset_dir.resolve()
    report = {
        "scope": "post hoc within-split exact-feature grouping; not an executable policy",
        "interpretation": "zero group gain concerns these feature maps and empirical pools, not all legal local histories",
        "splits": {}}
    for split in ("train", "validation", "test"):
        rows = json.loads((directory / f"{split}.json").read_text())["rows"]
        report["splits"][split] = {
            feature: audit(rows, feature) for feature in FEATURES}
    path = directory / "information_identifiability_audit.json"
    path.write_text(json.dumps(report, indent=2)+"\n")
    print(path)
    for split, result in report["splits"].items():
        print(split, result["response_features"])


if __name__ == "__main__":
    main()
