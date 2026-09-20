"""Resumable, seed-disjoint collection for the frozen downstream A4 teacher.

The split and source hashes are written before any labels are collected. No
sampling decision depends on SEND value or model performance.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parent
SOURCES = ("lookahead.py", "algorithm.py", "lookahead_policy.py",
           "lookahead_recipient_a4.py", "lookahead_a4_downstream.py",
           "physical_audit.py", "planning.py", "environment.py")


def source_hashes():
    return {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            for name in SOURCES}


def write_atomic(path, obj):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--start", type=int, default=982000)
    parser.add_argument("--train-seeds", type=int, default=180)
    parser.add_argument("--validation-seeds", type=int, default=45)
    parser.add_argument("--test-seeds", type=int, default=180)
    parser.add_argument("--shard-seeds", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-shards", type=int)
    parser.add_argument("--python", type=Path, default=ROOT.parent /
                        "external/envs/phmarl-py38/bin/python")
    parser.add_argument("--ages", type=int, nargs="+", default=(0, 8, 16))
    parser.add_argument("--include-acked-duplicate", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    splits = {}
    cursor = args.start
    for name, count in (("train", args.train_seeds),
                        ("validation", args.validation_seeds),
                        ("test", args.test_seeds)):
        splits[name] = {"first_seed": cursor, "seed_count": count}
        cursor += count
    manifest = {
        "scope": "frozen static look-ahead task; corrected physical A4 continuation",
        "selection": "prospective contiguous seed blocks; no SEND-value selection",
        "schedule_origin": args.start,
        "splits": splits,
        "shard_seeds": args.shard_seeds,
        "source_sha256": source_hashes(),
        "risk_states": ["safe", "risky"], "age_steps": args.ages,
        "overlap": "private direct overlap on every third seed",
        "acked_duplicate": ("prior delivery and ACK on every third seed (offset 1)"
                            if args.include_acked_duplicate else "none")}
    manifest_path = out / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("frozen A4 manifest/source hashes changed")
    else:
        write_atomic(manifest_path, manifest)
    jobs = []
    for split, info in splits.items():
        first = info["first_seed"]
        for offset in range(0, info["seed_count"], args.shard_seeds):
            count = min(args.shard_seeds, info["seed_count"]-offset)
            seed = first+offset
            tag = f"{split}_{seed}_{count}"
            jobs.append((split, seed, count, tag,
                         out / f"downstream_{tag}.json"))
    pending = [job for job in jobs if not job[4].exists()]
    if args.max_shards is not None:
        pending = pending[:args.max_shards]
    print(f"manifest={manifest_path} shards={len(jobs)} "
          f"pending={len(pending)} workers={args.workers}", flush=True)

    def run(job):
        split, seed, count, tag, path = job
        command = [str(args.python.resolve()), "-m", "coph_terrain.lookahead_a4_downstream",
                   "--start", str(seed), "--seeds", str(count), "--tag", tag,
                   "--schedule-origin", str(args.start), "--collection-only",
                   "--out-dir", str(out), "--ages", *map(str, args.ages)]
        if args.include_acked_duplicate:
            command.append("--include-acked-duplicate")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT.parent) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = "1"
        began = time.monotonic()
        result = subprocess.run(command, env=env, text=True,
                                capture_output=True)
        if result.returncode:
            raise RuntimeError(f"{tag}: {result.stderr[-3000:]}\n{result.stdout[-1000:]}")
        data = json.loads(path.read_text())
        expected = set(range(seed, seed+count))
        actual = {row["seed"] for row in data["rows"]}
        if actual != expected or data["failures"]:
            raise RuntimeError(f"{tag}: incomplete seed coverage or failures")
        return (split, tag, len(data["rows"]),
                sum(row["decision_value"] > 0 for row in data["rows"]),
                time.monotonic()-began)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, job): job for job in pending}
        for future in as_completed(futures):
            split, tag, n, positive, seconds = future.result()
            print(f"{split} {tag}: {n} rows, {positive} useful SEND, "
                  f"{seconds:.1f}s", flush=True)
    if any(not job[4].exists() for job in jobs):
        print("Partial collection saved; rerun the same command to resume.")
        return
    summary = {"manifest": str(manifest_path), "splits": {}}
    for split in splits:
        rows = []
        failures = []
        for name, _, _, _, path in jobs:
            if name != split:
                continue
            data = json.loads(path.read_text())
            rows.extend(data["rows"])
            failures.extend(data["failures"])
        seeds = {row["seed"] for row in rows}
        expected = set(range(splits[split]["first_seed"],
                             splits[split]["first_seed"] +
                             splits[split]["seed_count"]))
        if seeds != expected or failures:
            raise RuntimeError(f"{split}: incomplete collection")
        aggregate = out / f"{split}.json"
        write_atomic(aggregate, {"rows": rows, "failures": failures})
        summary["splits"][split] = {
            "file": str(aggregate), "n": len(rows),
            "positive_send": sum(row["decision_value"] > 0 for row in rows)}
    write_atomic(out / "collection_summary.json", summary)
    print(json.dumps(summary["splits"], indent=2), flush=True)


if __name__ == "__main__":
    main()
