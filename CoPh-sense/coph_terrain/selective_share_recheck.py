"""Compare forced-send and no-send continuations for the same acquired sets.

This is a restricted evaluator check of the communication choice, not a
learned A4 policy or a deployable joint oracle.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import AuditChoice, advance_without_acquisition, run_continuation


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def _one(row):
    if row["status"] != "evaluated":
        return {"parent_seed": row["parent_seed"], "requested_step":
                row["requested_step"], "status": row["status"]}
    env = CoPHTerrainEnv(row["family"], row["parent_seed"], 0,
                         seed=row["parent_seed"] + 9, executor="ph", device="cpu")
    advance_without_acquisition(env, row["requested_step"])
    choices = tuple(AuditChoice(item["agent"], item["modality"],
                                tuple(item["viewpoint"]),
                                tuple(item["target_region"])
                                if item.get("target_region") is not None else None)
                    for item in row["choices"])
    alternatives = []
    for selected, sent in zip(row["sets"], row["results"]):
        if not selected:
            continue
        held = run_continuation(env, choices, tuple(selected), transmit=False)
        alternatives.append({"selected": selected,
                             "send_cost": sent["score_single_world"],
                             "send_success": sent["success"],
                             "hold_cost": held["score_single_world"],
                             "hold_success": held["success"]})
    baseline = row["results"][0]
    feasible = ([(baseline["score_single_world"], "skip", ())]
                if baseline["success"] else [])
    for record in alternatives:
        for mode in ("send", "hold"):
            if record[f"{mode}_success"]:
                feasible.append((record[f"{mode}_cost"], mode,
                                 tuple(record["selected"])))
    best = min(feasible)
    return {"parent_seed": row["parent_seed"],
            "requested_step": row["requested_step"],
            "status": "evaluated", "baseline_cost": baseline["score_single_world"],
            "best_cost": best[0], "best_mode": best[1],
            "best_set": best[2], "alternatives": alternatives}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="joint_pose_fresh20_physical_v1.json")
    parser.add_argument("--tag", default="joint_pose_fresh20_selective_share_v1")
    args = parser.parse_args()
    source = OUT / args.source
    rows = json.loads(source.read_text())["records"]
    with ProcessPoolExecutor(max_workers=12) as pool:
        records = list(pool.map(_one, rows))
    evaluated = [row for row in records if row["status"] == "evaluated"]
    counts = {name: sum(row["best_mode"] == name for row in evaluated)
              for name in ("skip", "hold", "send")}
    by_size = {name: sum(len(row["best_set"]) == size
                         for row in evaluated)
               for name, size in (("skip", 0), ("single", 1), ("pair", 2))}
    path = OUT / f"{args.tag}.json"
    if path.exists():
        raise FileExistsError(path)
    summary = {"evaluated": len(evaluated), "best_mode": counts,
               "best_set_class": by_size,
               "scope": "min of scripted send/no-send, same physical acquisition sets"}
    path.write_text(json.dumps({"source": source.name, "summary": summary,
                                "records": records}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(path)


if __name__ == "__main__":
    main()
