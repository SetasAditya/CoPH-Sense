"""Frozen finite-prior material-pH conditional-scout gate."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .conditional_scout_benchmark import run_suite
from .conditional_scout_material import (
    DispatchHistory, FiniteTerrainPrior, candidate_scores,
    conditional_dispatch_teacher, delivery_causal_intervention,
    dispatch_candidates, evaluate_dispatch_critic, fit_dispatch_critic)
from .environment import CoPHTerrainEnv, TerrainConfig
from .physical_audit import advance_without_acquisition


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "conditional_scout_material_finite_v2"
THRESHOLDS = {"mean_regret": .02, "exact_rate": .90,
              "useful_recall": .90, "unnecessary_rejection": .90}


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True)+"\n")


def split_manifest(name, split_seed, states, prior_size, packet_seeds):
    rows = []
    for index in range(states):
        parent_seed = split_seed+101*index
        actual_seed = 1000+split_seed+17*index
        prior_seeds = [actual_seed]+[actual_seed+10000+j for j in range(prior_size-1)]
        rows.append({"state_id": f"{name}-{index:04d}", "family": "open",
                     "parent_seed": parent_seed, "actual_realization_seed": actual_seed,
                     "physics_seed": parent_seed+17,
                     "prefix_steps": 60+15*(index%4),
                     "prior_realization_seeds": prior_seeds,
                     "prior_weights": [1./prior_size]*prior_size,
                     "packet_seeds": list(packet_seeds)})
    return {"name": name, "seed": split_seed, "rows": rows}


def _collect_state(state):
        torch.set_num_threads(1)
        rows = []
        env = CoPHTerrainEnv(
            state["family"], state["parent_seed"], state["actual_realization_seed"],
            seed=state["physics_seed"], config=TerrainConfig(carrier_primary=True),
            executor="material_ph", device="cpu")
        advance_without_acquisition(env, state["prefix_steps"])
        if env.done:
            return [], {"state_id": state["state_id"], "status": "prefix_terminated"}
        history = DispatchHistory.from_env(env)
        prior = FiniteTerrainPrior.build(
            env, state["prior_realization_seeds"], state["prior_weights"])
        candidates = dispatch_candidates(env)
        for candidate_index, candidate in enumerate(candidates):
            label = conditional_dispatch_teacher(
                history, prior, candidate, tuple(state["packet_seeds"]))
            value = label["value"]
            if not np.isfinite(value): value = -1e6
            row = {"state_id": state["state_id"], "candidate_index": candidate_index,
                   "candidate_id": candidate.candidate_id,
                   "features": list(candidate.features), "feasible": candidate.feasible,
                   "heuristic_score": candidate.heuristic_score,
                   "unknown_fraction": candidate.unknown_fraction,
                   "value": float(value), "stderr": label["stderr"],
                   "interval": list(label["interval"]),
                   "classification": label["classification"],
                   "posterior": label["posterior"],
                   "world_values": [branch["value"] for branch in label["branches"]]}
            rows.append(row)
        classes = [row["classification"] for row in rows[1:]]
        return rows, {"state_id": state["state_id"], "status": "labeled",
                       "candidate_count": len(rows),
                       "useful": classes.count("useful"),
                       "unnecessary": classes.count("unnecessary"),
                       "unresolved": classes.count("unresolved"),
                       "infeasible": classes.count("infeasible")}


def _collect_cached(arguments):
    state, cache_dir = arguments
    path = Path(cache_dir)/f"{state['state_id']}.json"
    if path.exists():
        payload = json.loads(path.read_text())
        return payload["rows"], payload["audit"]
    block = _collect_state(state)
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, {"rows": block[0], "audit": block[1]})
    return block


def collect(manifest, workers=1, cache_dir=None):
    cache_dir = Path(cache_dir or ".dispatch_cache")
    arguments = [(state, str(cache_dir)) for state in manifest["rows"]]
    if workers <= 1:
        # Recreate the worker per state. VMAS/Torch retain native allocations
        # after a large counterfactual block; process recycling bounds memory.
        blocks = []
        for item in arguments:
            with ProcessPoolExecutor(max_workers=1) as executor:
                blocks.append(next(executor.map(_collect_cached, (item,))))
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as executor:
            blocks = list(executor.map(_collect_cached, arguments))
    rows = [row for block, _ in blocks for row in block]
    states = [audit for _, audit in blocks]
    return rows, states


def _group(rows):
    grouped = {}
    for row in rows: grouped.setdefault(row["state_id"], []).append(row)
    return {key: sorted(value, key=lambda row: row["candidate_index"])
            for key, value in grouped.items()}


def baseline_report(rows, learned=None):
    result = {}
    for name in ("idle", "always", "heuristic", "oracle", "learned"):
        if name == "learned" and learned is None: continue
        regrets, costs, exact = [], [], []
        for candidates in _group(rows).values():
            values = np.asarray([row["value"] for row in candidates])
            optimal = set(np.flatnonzero(values >= values.max()-1e-8))
            feasible = [i for i, row in enumerate(candidates[1:], 1) if row["feasible"]]
            if name == "idle": chosen = 0
            elif name == "always": chosen = feasible[0] if feasible else 0
            elif name == "oracle": chosen = int(np.argmax(values))
            elif name == "heuristic":
                # Frozen public proxy: uncertainty minus sensing and normalized travel time.
                scores = [0.]+[row["heuristic_score"] if row["feasible"] else -np.inf
                               for row in candidates[1:]]
                chosen = int(np.argmax(scores))
            else:
                pseudo = [type("C", (), {"features": tuple(row["features"]),
                                           "feasible": row["feasible"]})
                          for row in candidates]
                chosen = int(np.argmax(candidate_scores(learned, pseudo)))
            regrets.append(float(values.max()-values[chosen]))
            costs.append(float(-values[chosen])); exact.append(chosen in optimal)
        result[name] = {"mean_regret": float(np.mean(regrets)),
                        "mean_relative_cost": float(np.mean(costs)),
                        "exact_rate": float(np.mean(exact)), "states": len(regrets)}
    return result


def conflicting_input_audit(rows, decimals=7):
    groups = {}
    for row in rows:
        key = tuple(np.round(row["features"], decimals))
        groups.setdefault(key, set()).add(round(row["value"], 7))
    return {"groups": len(groups),
            "conflicting_groups": sum(len(values)>1 for values in groups.values())}


def geometric_reconciliation(out):
    physical_file = HERE/"results"/"conditional_scout_geometric_v1"/"physical_rollouts.csv"
    physical = list(csv.DictReader(physical_file.open())) if physical_file.exists() else []
    analytic = {row["name"]: row for row in run_suite()}
    report = {"historical_artifacts_modified": False,
              "analytic_objective": "declared route cost plus dispatch cost",
              "physical_visual_score": "time + 0.10 carrier motion + 0.06 scout motion + dispatch cost + collision + dynamic pulse",
              "route_cost_in_physical_score": False, "cases": []}
    for case, row in analytic.items():
        subset = [item for item in physical if item["case"] == case]
        report["cases"].append({"case": case, "analytic_idle_cost": row["idle_cost"],
            "analytic_conditional_cost": row["conditional_cost"],
            "physical_rows": len(subset),
            "mean_reported_true_route_cost": (float(np.mean([
                float(item.get("true_route_cost", "nan")) for item in subset]))
                if subset and "true_route_cost" in subset[0] else None)})
    _write(out/"geometric_objective_reconciliation.json", report)
    return report


def gate_decision(test, coverage, canonical, causal, complete_useful):
    checks = {
        "coverage": coverage["useful"] > 0 and coverage["unnecessary"] > 0,
        "mean_regret": test.get("mean_regret", np.inf) < THRESHOLDS["mean_regret"],
        "exact_rate": test.get("exact_rate", 0.) >= THRESHOLDS["exact_rate"],
        "useful_recall": (test.get("useful_denominator", 0) > 0 and
                          test.get("useful_recall", 0.) >= THRESHOLDS["useful_recall"]),
        "unnecessary_rejection": (test.get("unnecessary_denominator", 0) > 0 and
            test.get("unnecessary_rejection", 0.) >= THRESHOLDS["unnecessary_rejection"]),
        "canonical_useful_known_late": all(canonical.get(name, False)
                                            for name in ("useful", "known", "too_late")),
        "causal_delivery": bool(causal.get("passed", False)),
        "complete_dispatch_benefit": bool(complete_useful),
        "recovery": bool(causal.get("delivered", {}).get("recovered", False)
                         and causal.get("withheld", {}).get("recovered", False)),
    }
    return {"passed": all(checks.values()), "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--admission-states", type=int, default=6)
    parser.add_argument("--train-states", type=int, default=24)
    parser.add_argument("--val-states", type=int, default=8)
    parser.add_argument("--test-states", type=int, default=12)
    parser.add_argument("--prior-size", type=int, default=4)
    parser.add_argument("--packet-replicates", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    packet_seeds = tuple(range(args.packet_replicates))
    specifications = (("admission", 11000, args.admission_states),
                      ("train", 21000, args.train_states),
                      ("validation", 31000, args.val_states),
                      ("test", 41000, args.test_states))
    manifests, datasets, state_audits = {}, {}, {}
    for name, seed, count in specifications:
        manifest = split_manifest(name, seed, count, args.prior_size, packet_seeds)
        manifests[name] = manifest; _write(args.out/f"{name}_manifest.json", manifest)
    if args.resume and (args.out/"admission.json").exists():
        rows = json.loads((args.out/"admission.json").read_text())
        audit = json.loads((args.out/"admission_audit.json").read_text())
    else:
        rows, audit = collect(manifests["admission"], args.workers,
                              args.out/"cache"/"admission")
    datasets["admission"], state_audits["admission"] = rows, audit
    _write(args.out/"admission.json", rows); _write(args.out/"admission_audit.json", audit)
    admission_classes = [row["classification"] for row in datasets["admission"]
                         if row["candidate_id"] != "idle"]
    coverage = {name: admission_classes.count(name) for name in
                ("useful", "unnecessary", "unresolved", "infeasible")}
    record = {"scope": "restricted finite-prior gate; not final E2",
              "thresholds": THRESHOLDS, "admission_coverage": coverage,
              "trained": False, "gate": "blocked_admission_coverage",
              "geometric_reconciliation": geometric_reconciliation(args.out),
              "source_sha256": {name: _sha(HERE/name) for name in (
                  "conditional_scout_material.py", "conditional_scout.py",
                  "environment.py", "run_conditional_scout_material.py")}}
    if coverage["useful"] and coverage["unnecessary"]:
        for name in ("train", "validation", "test"):
            rows, audit = collect(manifests[name], args.workers,
                                  args.out/"cache"/name)
            datasets[name], state_audits[name] = rows, audit
            _write(args.out/f"{name}.json", rows)
            _write(args.out/f"{name}_audit.json", audit)
        record["input_audit"] = conflicting_input_audit(datasets["train"])
        model, validation = fit_dispatch_critic(
            datasets["train"], datasets["validation"], epochs=args.epochs)
        test = evaluate_dispatch_critic(model, datasets["test"])
        torch.save(model.state_dict(), args.out/"dispatch_value.pt")
        baselines = baseline_report(datasets["test"], model)
        admission_nonidle = [row for row in datasets["admission"]
                             if row["candidate_id"] != "idle"]
        # Predeclared semantic predicates, evaluated independently of model.
        canonical = {
            "useful": any(row["classification"] == "useful"
                          for row in admission_nonidle),
            "known": any(row["unknown_fraction"] <= .05
                         and row["classification"] == "unnecessary"
                         for row in admission_nonidle),
            "too_late": any(row["classification"] == "infeasible"
                            for row in admission_nonidle),
        }
        causal = {"passed": False, "reason": "learned_policy_selected_idle"}
        complete_useful = False
        development = _group(datasets["admission"])
        for state_id, rows in development.items():
            scores = np.full(len(rows), -np.inf); scores[0] = 0.
            feasible = [i for i, row in enumerate(rows[1:], 1) if row["feasible"]]
            if feasible:
                x = torch.tensor([rows[i]["features"] for i in feasible], dtype=torch.float32)
                with torch.inference_mode(): scores[feasible] = model(x).numpy()
            chosen = int(np.argmax(scores))
            if chosen != 0:
                spec = next(item for item in manifests["admission"]["rows"]
                            if item["state_id"] == state_id)
                env = CoPHTerrainEnv(spec["family"], spec["parent_seed"],
                    spec["actual_realization_seed"], seed=spec["physics_seed"],
                    config=TerrainConfig(carrier_primary=True),
                    executor="material_ph", device="cpu")
                advance_without_acquisition(env, spec["prefix_steps"])
                candidate = dispatch_candidates(env)[chosen]
                causal = delivery_causal_intervention(env, candidate)
                complete_useful = rows[chosen]["interval"][0] > 0
                break
        gate = gate_decision(test, coverage, canonical, causal, complete_useful)
        record.update({"trained": True, "validation": validation, "test": test,
                       "baselines": baselines, "canonical": canonical,
                       "causal": causal, "gate_details": gate,
                       "gate": "passed" if gate["passed"] else "failed_complete_gate"})
    _write(args.out/"gate.json", record)
    (args.out/"gate.sha256").write_text(_sha(args.out/"gate.json")+"\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__": main()
