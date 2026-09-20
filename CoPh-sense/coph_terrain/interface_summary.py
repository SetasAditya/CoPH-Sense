"""Summarize the disjoint subset of the interface audit.

Parents 850000--850011 were previously used by the cap-stress diagnostic;
exclude them from the fresh-result summary without altering raw artifacts.
"""

import json
from pathlib import Path
from statistics import median


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def main():
    physical = json.loads((OUT / "interface_revision_excluded40_physical_v1.json").read_text())
    funnel = json.loads((OUT / "interface_revision_excluded40_value_funnel_v1.json").read_text())
    p_rows = [row for row in physical["records"]
              if row["parent_seed"] >= 850012]
    f_rows = [row for row in funnel["records"] if row["seed"] >= 850012]
    p_evaluated = [row for row in p_rows if row["status"] == "evaluated"]
    f_evaluated = [row for row in f_rows if row["status"] == "evaluated"]
    selected = [row for row in p_evaluated if row["best_set"] is not None]
    high = [row for row in f_evaluated if row["top"]["V_GT"] > .05]
    result = {
        "fresh_parent_range": [850012, 850039],
        "excluded_reused_parent_range": [850000, 850011],
        "physical": {
            "scheduled": len(p_rows), "evaluated": len(p_evaluated),
            "covered": len(selected) + sum(
                row["status"] != "evaluated" and row["prefix_success"]
                for row in p_rows),
            "best_set_class": {
                name: sum(len(row["best_set"]) == size for row in selected)
                for name, size in (("skip", 0), ("single", 1), ("pair", 2))},
        },
        "value_funnel": {
            "scheduled": len(f_rows), "evaluated": len(f_evaluated),
            "high_ideal_G": sum(any(region["V_G"] > .05 for region in row["regions"])
                                for row in f_evaluated),
            "high_ideal_T": sum(any(region["V_T"] > .05 for region in row["regions"])
                                for row in f_evaluated),
            "high_ideal_GT": len(high),
            "strong_ideal_interaction": sum(any(
                region["S_GT"] > .05 and region["V_GT"] > .05 and
                region["all_success"] for region in row["regions"])
                for row in f_evaluated),
            "high_top_menu_G": sum(row["top"]["menu_G"] for row in high),
            "high_top_menu_T": sum(row["top"]["menu_T"] for row in high),
            "high_top_actual_GT_positive": sum(
                row["tiers"]["GT"]["actual_value"] > .01 for row in high),
            "high_top_dwell_GT_positive": sum(
                row["tiers"]["GT"]["dwell_value"] > .01 for row in high),
            "high_top_physical_GT_positive": sum(
                row["tiers"]["GT"]["physical_value"] > .01 for row in high),
            "median_values": {
                "ideal": median(row["top"]["V_GT"] for row in high),
                "actual": median(row["tiers"]["GT"]["actual_value"] for row in high),
                "dwell": median(row["tiers"]["GT"]["dwell_value"] for row in high),
                "physical": median(row["tiers"]["GT"]["physical_value"] for row in high),
            },
        },
        "scope": "single-world restricted scripted pH continuation, not belief Q*",
    }
    path = OUT / "interface_revision_fresh28_summary_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
