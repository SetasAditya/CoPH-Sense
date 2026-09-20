#!/usr/bin/env python3
"""Freeze an oracle-selected layout before querying frozen A5 on it."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from coph_fork.search_two_fork_physical_family import OUT


HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    selected_path = OUT / "layout_01_geometry.json"
    full_path = OUT / "layout_01_full_seven_action_oracle.json"
    pricing_path = OUT / "layout_01_geometry_local_pricing.json"
    neighbor_path = OUT / "layout_deep_geometry.json"
    selected = json.loads(selected_path.read_text())
    full = json.loads(full_path.read_text())
    pricing = json.loads(pricing_path.read_text())
    neighbor = json.loads(neighbor_path.read_text())
    local = next(row for row in pricing["results"]
                 if row["safe_prior_1"] == .65
                 and row["safe_prior_2"] == .60
                 and row["sensing_cost"] == .02)
    if full["best"] != {"M0": "R1", "M1": "R2"}:
        raise RuntimeError("seven-action physical oracle did not select R1 then R2")
    if not full["all_success"] or full["max_collision_cost"] != 0:
        raise RuntimeError("physical continuations failed")
    if not local["summary"]["passed"] or not neighbor["summary"]["passed"]:
        raise RuntimeError("margin or geometric robustness check failed")
    if not all(row["summary"]["passed"] for row in pricing["results"]):
        raise RuntimeError("nearby prior/cost perturbation failed")
    artifacts = [selected_path, full_path, pricing_path, neighbor_path]
    sources = [HERE / name for name in (
        "two_fork_scenario.py", "two_fork_environment.py",
        "two_fork_motion.py", "search_two_fork_physical_family.py")]
    frozen = {
        "stage": "physical-oracle-only map selection; before A5 evaluation",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "layout": full["layout"],
        "horizon": 2600,
        "communication_delay_steps": 3,
        "packet_drop_probability": 0.,
        "candidate_actions": ["skip", "R1_G", "R1_T", "R1",
                              "R2_G", "R2_T", "R2"],
        "q": full["q"], "best": full["best"],
        "required_margin": .05,
        "margins": local["summary"]["contrast"],
        "local_perturbations_passing": len(pricing["results"]),
        "local_perturbations_total": len(pricing["results"]),
        "neighbor_layout": neighbor["layout"],
        "neighbor_margins": neighbor["summary"]["contrast"],
        "scope": "restricted scripted VMAS carrier continuation from matched pre-Fork-1 snapshots; development family, not held-out generalization",
        "sha256": {str(path.relative_to(HERE)): digest(path) for path in artifacts + sources},
    }
    path = OUT / "frozen_physical_layout_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(frozen, indent=2) + "\n")
    print(path)
    print("frozen physical best", frozen["best"],
          "minimum margin", min(frozen["margins"].values()))


if __name__ == "__main__":
    main()
