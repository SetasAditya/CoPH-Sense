"""Resumable full-test raw-broadcast comparator for the fixed A4 source pool."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .lookahead_a4_learning import augment_response


ROOT = Path(__file__).resolve().parent


def worker(source, output):
    rows = json.loads(source.read_text())["rows"]
    augment_response(rows, raw_broadcast=True)
    slim = [{key: row[key] for key in
             ("seed", "risk", "age_steps", "receiver_condition",
              "raw_send", "raw_bytes", "raw_lookahead_gain_m")}
            for row in rows]
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"rows": slim}, separators=(",", ":"))+"\n")
    os.replace(temporary, output)
    print(output, len(slim), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_dir", type=Path, nargs="?")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--python", type=Path, default=ROOT.parent /
                        "external/envs/phmarl-py38/bin/python")
    parser.add_argument("--worker-input", type=Path)
    parser.add_argument("--worker-output", type=Path)
    parser.add_argument("--partial", action="store_true")
    args = parser.parse_args()
    if args.worker_input:
        worker(args.worker_input, args.worker_output)
        return
    directory = args.dataset_dir.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    test = manifest["splits"]["test"]
    first = test["first_seed"]
    count = test["seed_count"]
    shard_size = manifest["shard_seeds"]
    raw = directory / "raw_broadcast"
    raw.mkdir(exist_ok=True)
    jobs = []
    for offset in range(0, count, shard_size):
        seed = first+offset
        n = min(shard_size, count-offset)
        source = directory / f"downstream_test_{seed}_{n}.json"
        if not source.exists():
            if args.partial:
                continue
            raise FileNotFoundError(source)
        target = raw / f"raw_test_{seed}_{n}.json"
        jobs.append((source, target))
    raw_manifest = {
        "source_manifest_sha256": hashlib.sha256(
            (directory / "manifest.json").read_bytes()).hexdigest(),
        "raw_collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "replay_sha256": hashlib.sha256(
            (ROOT / "lookahead_a4_learning.py").read_bytes()).hexdigest(),
        "test_first_seed": first, "test_seed_count": count}
    path = raw / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != raw_manifest:
        raise ValueError("raw-broadcast source changed")
    if not path.exists():
        path.write_text(json.dumps(raw_manifest, indent=2)+"\n")

    def run(job):
        source, target = job
        if target.exists():
            return target
        command = [str(args.python.resolve()), "-m",
                   "coph_terrain.lookahead_a4_raw_collect",
                   "--worker-input", str(source),
                   "--worker-output", str(target)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT.parent) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            env[name] = "1"
        result = subprocess.run(command, env=env, text=True,
                                capture_output=True)
        if result.returncode:
            raise RuntimeError(f"{source}: {result.stderr[-2000:]}")
        return target

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for future in as_completed(executor.submit(run, job) for job in jobs):
            print(future.result(), flush=True)
    if args.partial:
        print("partial raw comparator", len(jobs), "available shards")
        return
    expected = [row for job in jobs for row in
                json.loads(job[0].read_text())["rows"]]
    raw_rows = [row for job in jobs for row in
                json.loads(job[1].read_text())["rows"]]
    keys = ("seed", "risk", "age_steps", "receiver_condition")
    if len(expected) != len(raw_rows) or any(
        tuple(a[k] for k in keys) != tuple(b[k] for k in keys)
        for a, b in zip(expected, raw_rows)):
        raise ValueError("raw comparator does not match compact source rows")
    print("raw comparator complete", len(raw_rows))


if __name__ == "__main__":
    main()
