"""Paired, resumable six-method confirmatory evaluation campaign.

This module owns experiment scheduling and reporting, not policy behavior.  An
episode adapter receives an :class:`EpisodeKey` and a method name and must
return the declared metrics.  Keeping this boundary narrow makes it impossible
for different baselines to silently receive different maps or random seeds.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import hashlib
import importlib
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .evaluation import empirical_cvar


METHODS = ("independent", "raw_broadcast", "compact_broadcast", "novelty_mi",
           "need_request_match", "full_coph")
REQUIRED_METRICS = ("success", "team_cost", "material_risk", "bytes")
OPTIONAL_METRICS = ("duplicate_sensing", "disjoint_support", "lookahead_gain")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EpisodeKey:
    split: str
    training_seed: int
    group_id: str
    family: str
    parent_seed: int
    realization: int
    episode_seed: int

    @property
    def block_id(self):
        return (self.split, self.training_seed, self.group_id, self.realization)


def episode_seed(training_seed, parent_seed, realization):
    """Stable seed independent of method and Python's randomized hash."""
    raw = f"{training_seed}:{parent_seed}:{realization}:coph-e2-v1".encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:4], "little")


def load_schedule(manifest_paths, training_seeds, limit_groups=None,
                  realizations=None):
    keys = []
    for path in map(Path, manifest_paths):
        payload = json.loads(path.read_text())
        split = payload["split"]
        groups = payload["groups"]
        if limit_groups is not None:
            groups = groups[:limit_groups]
        count = (payload.get("episode_realizations_per_test_group") or 5
                 if realizations is None else realizations)
        for seed in training_seeds:
            for group in groups:
                for realization in range(count):
                    keys.append(EpisodeKey(
                        split, int(seed), group["group_id"], group["family"],
                        int(group["parent_seed"]), realization,
                        episode_seed(seed, group["parent_seed"], realization)))
    return keys


def _number(value, name):
    if isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def validate_metrics(metrics):
    missing = set(REQUIRED_METRICS)-set(metrics)
    if missing:
        raise ValueError(f"missing metrics: {sorted(missing)}")
    success = metrics["success"]
    if success not in (False, True, 0, 1):
        raise ValueError("success must be binary")
    clean = {"success": bool(success)}
    for name in REQUIRED_METRICS[1:]:
        clean[name] = _number(metrics[name], name)
        if clean[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in OPTIONAL_METRICS:
        value = metrics.get(name)
        clean[name] = None if value is None else _number(value, name)
        if clean[name] is not None and clean[name] < 0:
            raise ValueError(f"{name} must be nonnegative")
    return clean


def record_id(key, method):
    return "/".join(map(str, (*key.block_id, method)))


def run_campaign(keys, evaluator: Callable[[EpisodeKey, str], dict], output,
                 methods=METHODS, resume=True):
    """Evaluate complete paired blocks and append durable JSONL records."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    present = set()
    if resume and output.exists():
        for line in output.read_text().splitlines():
            if line.strip():
                present.add(json.loads(line)["record_id"])
    with output.open("a") as stream:
        for key in keys:
            for method in methods:
                rid = record_id(key, method)
                if rid in present:
                    continue
                metrics = validate_metrics(evaluator(key, method))
                row = {"schema_version": SCHEMA_VERSION, "record_id": rid,
                       "method": method, **asdict(key), **metrics}
                stream.write(json.dumps(row, sort_keys=True)+"\n")
                stream.flush()
                present.add(rid)
    return len(present)


def load_records(path, methods=METHODS, require_complete=True):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip()]
    seen = set()
    blocks = {}
    for row in rows:
        if row["schema_version"] != SCHEMA_VERSION or row["method"] not in methods:
            raise ValueError("unknown schema or method")
        if row["record_id"] in seen:
            raise ValueError(f"duplicate record {row['record_id']}")
        seen.add(row["record_id"])
        key = (row["split"], row["training_seed"], row["group_id"],
               row["realization"])
        blocks.setdefault(key, {})[row["method"]] = row
    if require_complete:
        incomplete = {key: sorted(set(methods)-set(value))
                      for key, value in blocks.items() if set(value) != set(methods)}
        if incomplete:
            raise ValueError(f"incomplete paired blocks: {incomplete}")
    return rows


def aggregate(records, methods=METHODS, cvar_alpha=.95):
    report = {"schema_version": SCHEMA_VERSION, "cvar_alpha": cvar_alpha,
              "methods": {}, "paired_blocks": len(records)//len(methods)}
    for split in sorted({row["split"] for row in records}):
        section = {}
        split_rows = [row for row in records if row["split"] == split]
        for method in methods:
            items = [row for row in split_rows if row["method"] == method]
            values = {"episodes": len(items),
                      "success": float(np.mean([x["success"] for x in items])),
                      "team_cost": float(np.mean([x["team_cost"] for x in items])),
                      "material_risk": float(np.mean([x["material_risk"] for x in items])),
                      "risk_cvar_95": empirical_cvar(
                          [x["material_risk"] for x in items], cvar_alpha),
                      "bytes": float(np.mean([x["bytes"] for x in items]))}
            for name in OPTIONAL_METRICS:
                available = [x[name] for x in items if x.get(name) is not None]
                values[name] = (None if not available else float(np.mean(available)))
            section[method] = values
        report["methods"][split] = section
    report["paired_full_coph_effects"] = {}
    rng = np.random.default_rng(2027)
    for split in report["methods"]:
        split_rows = [row for row in records if row["split"] == split]
        blocks = {}
        for row in split_rows:
            key = (row["training_seed"], row["group_id"], row["realization"])
            blocks.setdefault(key, {})[row["method"]] = row
        effects = {}
        for method in methods:
            if method == "full_coph":
                continue
            metric_rows = {}
            for metric in ("success", "team_cost", "material_risk", "bytes"):
                differences = np.asarray([
                    float(block["full_coph"][metric])-float(block[method][metric])
                    for block in blocks.values()], dtype=np.float64)
                draws = np.asarray([rng.choice(differences, len(differences),
                                               replace=True).mean()
                                    for _ in range(2000)])
                metric_rows[metric] = {
                    "full_coph_minus_rival": float(differences.mean()),
                    "paired_block_bootstrap_95_interval": [
                        float(np.quantile(draws, .025)),
                        float(np.quantile(draws, .975))]}
            effects[method] = metric_rows
        report["paired_full_coph_effects"][split] = effects
    return report


def write_report(report, directory, label="confirmatory"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/f"{label}_summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)+"\n")
    columns = ("split", "method", "episodes", "success", "team_cost",
               "risk_cvar_95", "material_risk", "bytes", *OPTIONAL_METRICS)
    with (directory/f"{label}_table.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for split, methods in report["methods"].items():
            for method, values in methods.items():
                writer.writerow({"split": split, "method": method, **values})
    lines = ["# Confirmatory campaign summary", "", "Paired blocks: " +
             str(report["paired_blocks"]), ""]
    for split, methods in report["methods"].items():
        lines += [f"## {split}", "",
                  "| Method | Success | Team cost | Risk CVaR.95 | Bytes | Duplicate | Disjoint | Look-ahead |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for method, v in methods.items():
            fmt = lambda x: "--" if x is None else f"{x:.4f}"
            lines.append(f"| {method} | {v['success']:.4f} | {v['team_cost']:.4f} | "
                         f"{v['risk_cvar_95']:.4f} | {v['bytes']:.1f} | "
                         f"{fmt(v['duplicate_sensing'])} | {fmt(v['disjoint_support'])} | "
                         f"{fmt(v['lookahead_gain'])} |")
        lines.append("")
    (directory/f"{label}_table.md").write_text("\n".join(lines)+"\n")
    try:
        import matplotlib.pyplot as plt
        for split, methods in report["methods"].items():
            names = list(methods)
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
            for ax, metric, title in zip(
                    axes, ("team_cost", "risk_cvar_95", "bytes"),
                    ("Team cost ↓", "Risk CVaR.95 ↓", "Charged bytes ↓")):
                ax.bar(np.arange(len(names)), [methods[n][metric] for n in names])
                ax.set_title(title); ax.set_xticks(np.arange(len(names)))
                ax.set_xticklabels(names, rotation=35, ha="right", fontsize=7)
            fig.tight_layout()
            fig.savefig(directory/f"{label}_{split}.png", dpi=180)
            plt.close(fig)
    except ImportError:
        pass


def source_manifest(paths, extra=None):
    rows = {str(Path(path)): hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in paths}
    payload = {"schema_version": SCHEMA_VERSION, "methods": METHODS,
               "source_sha256": rows, "extra": extra or {}}
    payload["manifest_sha256"] = hashlib.sha256(json.dumps(
        payload, sort_keys=True).encode()).hexdigest()
    return payload


def _load_adapter(spec):
    module, function = spec.split(":", 1)
    return getattr(importlib.import_module(module), function)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--adapter", required=True,
                        help="Python module:function episode adapter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--realizations", type=int)
    args = parser.parse_args()
    keys = load_schedule(args.manifests, args.training_seeds,
                         args.limit_groups, args.realizations)
    run_campaign(keys, _load_adapter(args.adapter), args.output)
    records = load_records(args.output)
    write_report(aggregate(records), args.output.parent)
    here = Path(__file__).resolve().parent
    sources = [here/name for name in (
        "confirmatory_campaign.py", "evaluation.py", "environment.py",
        "material_executor.py", "generator.py", "scenario.py")]
    sources += list(args.manifests)
    for optional in (here/"results/material_ph_calibration_v1.json",
                     here/"results/lookahead_bidirectional_a4/frozen_learning_manifest.json"):
        if optional.exists():
            sources.append(optional)
    protocol = source_manifest(sources, {
        "adapter": args.adapter,
        "training_seeds": args.training_seeds,
        "limit_groups": args.limit_groups,
        "realizations": args.realizations,
        "status": ("engineering_smoke" if "smoke" in args.adapter
                   else "confirmatory")})
    (args.output.parent/"protocol_manifest.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
