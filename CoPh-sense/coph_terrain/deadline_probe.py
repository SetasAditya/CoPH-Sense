"""Excluded-parent diagnosis of no-success restricted-audit states."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import json
from pathlib import Path

from .environment import CoPHTerrainEnv
from .physical_audit import advance_without_acquisition, run_continuation


OUT = Path(__file__).resolve().parent / "results" / "pilot"


def probe(record):
    seed = record["parent_seed"]
    env = CoPHTerrainEnv(record["family"], seed, 0, seed=seed + 9,
                         executor="ph", device="cpu")
    advance_without_acquisition(env, record["decision_step"])
    env.config = replace(env.config, horizon_steps=1800)
    result = run_continuation(env, (), ())
    original = record["results"][0]
    return {"family": record["family"], "parent_seed": seed,
            "original_60s": original,
            "empty_set_90s": result}


def main():
    source = OUT / "resource_audit_excluded200_v1.json"
    records = json.loads(source.read_text())["records"]
    targets = [record for record in records if record["no_success_option"]]
    with ProcessPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(probe, targets))
    summary = {"parents": len(results),
               "original_empty_success": sum(
                   row["original_60s"]["success"] for row in results),
               "empty_success_at_90s": sum(
                   row["empty_set_90s"]["success"] for row in results),
               "by_family": {family: {
                   "parents": sum(row["family"] == family for row in results),
                   "empty_success_at_90s": sum(
                       row["family"] == family and row["empty_set_90s"]["success"]
                       for row in results)}
                   for family in ("open", "bottleneck", "labyrinth")},
               "scope": "diagnostic skip continuation only; not a rerun of all sensing sets"}
    output = {"schema_version": 1, "source": str(source),
              "summary": summary, "records": results}
    path = OUT / "deadline_probe_no_success_90s_v1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print(path)


if __name__ == "__main__":
    main()
