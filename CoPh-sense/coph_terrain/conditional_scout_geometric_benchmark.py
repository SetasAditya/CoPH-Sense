"""Frozen analytic-versus-physical benchmark for the conditional scout.

The analytic teacher and the geometric rollout intentionally retain their
original objectives.  This runner reports both and flags disagreements; it
does not tune either objective from held-out outcomes.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .conditional_scout_benchmark import default_cases, evaluate_case
from .conditional_scout_learning import evaluate_model, train_model
from .conditional_scout_long_horizon import run_rollout


HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "results" / "conditional_scout_geometric_v1"
POLICIES = ("never", "always", "oracle", "learned")


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(args):
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    learning = out / "learning"
    checkpoint, _ = train_model(
        learning, n_train=args.n_train, n_val=args.n_val,
        epochs=args.epochs, seed=args.train_seed, device=args.device)
    analytic = evaluate_model(
        checkpoint, out / "analytic_test", n_test=args.n_test,
        seed=args.test_seed, device=args.device)

    cases = default_cases()
    rows = []
    for case_index, case in enumerate(cases):
        for mode_index, mode in enumerate(case.modes):
            seed = args.rollout_seed + 100 * case_index + mode_index
            for policy in POLICIES:
                metrics = run_rollout(
                    case.name, policy=policy,
                    ckpt=checkpoint if policy == "learned" else None,
                    outdir=out / "rollouts" / case.name / mode.name / policy,
                    seed=seed, max_steps=args.max_steps, device=args.device,
                    frame_stride=args.frame_stride, mode_name=mode.name,
                    render=False)
                rows.append(metrics)

    # Compare the analytic ex-ante decision with the expected executed score
    # over the same declared hidden-mode probabilities.
    discrepancy = []
    for case in cases:
        subset = [row for row in rows if row["case"] == case.name]
        weights = {mode.name: mode.probability / sum(m.probability for m in case.modes)
                   for mode in case.modes}
        expected = {}
        for policy in POLICIES:
            expected[policy] = float(sum(
                weights[row["mode"]] * row["visual_score"]
                for row in subset if row["policy"] == policy))
        physical_dispatch = expected["always"] < expected["never"]
        analytic_row = evaluate_case(case)
        discrepancy.append({
            "case": case.name,
            "analytic_dispatch": bool(analytic_row["dispatch"]),
            "physical_dispatch_preferred": bool(physical_dispatch),
            "analytic_value": float(analytic_row["scout_value"]),
            "expected_physical_score": expected,
            "analytic_physical_disagree":
                bool(analytic_row["dispatch"] != physical_dispatch),
        })

    # Render only representative useful dynamic rollouts.  Every policy/mode
    # was evaluated above from paired seeds; GIF generation is presentation-only.
    for policy in ("never", "oracle", "learned"):
        run_rollout(
            "moving_blocker_prediction", policy=policy,
            ckpt=checkpoint if policy == "learned" else None,
            outdir=out / "representative" / policy,
            seed=args.rollout_seed, max_steps=args.max_steps,
            device=args.device, frame_stride=args.frame_stride,
            mode_name="clearing", render=True)

    fields = ("case", "mode", "policy", "dispatch", "predicted_value",
              "reached_goal", "collision", "blocked_by_safety", "time",
              "carrier_path_length", "scout_path_length", "report_time",
              "report_received_timely", "route_changed_due_to_report",
              "scout_recovered_at_end", "min_carrier_clearance",
              "min_scout_clearance", "visual_score")
    with (out / "physical_rollouts.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)

    manifest = {
        "scope": "frozen Yiwang geometric benchmark; not VMAS or material-pH",
        "train_seed": args.train_seed, "test_seed": args.test_seed,
        "rollout_seed": args.rollout_seed,
        "n_train": args.n_train, "n_val": args.n_val,
        "n_test": args.n_test, "epochs": args.epochs,
        "checkpoint": str(Path(checkpoint).relative_to(out)),
        "checkpoint_sha256": _sha(checkpoint),
        "source_sha256": {
            name: _sha(HERE / name) for name in (
                "conditional_scout_benchmark.py",
                "conditional_scout_learning.py",
                "conditional_scout_long_horizon.py")},
    }
    result = {
        "manifest": manifest, "analytic": analytic,
        "physical_rollout_count": len(rows),
        "physical_success_rate": float(np.mean([r["reached_goal"] for r in rows])),
        "physical_collision_rate": float(np.mean([r["collision"] for r in rows])),
        "recovery_rate_when_dispatched": float(np.mean([
            r["scout_recovered_at_end"] for r in rows if r["dispatch"]])),
        "discrepancies": discrepancy,
    }
    (out / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Conditional-scout geometric benchmark", "",
        "This is a kinematic geometric diagnostic. It is not a VMAS/material-pH result.", "",
        f"Analytic held-out count: {analytic['count']}",
        f"Analytic dispatch accuracy: {analytic['dispatch_accuracy']:.4f}",
        f"Analytic regret to oracle: {analytic['mean_learned_regret_to_oracle']:.6f}",
        f"Physical paired rollouts: {len(rows)}",
        f"Physical success rate: {result['physical_success_rate']:.4f}",
        f"Physical collision rate: {result['physical_collision_rate']:.4f}", "",
        "| Case | Analytic dispatch | Physical dispatch preferred | Disagree |",
        "|---|---:|---:|---:|",
    ]
    for row in discrepancy:
        lines.append(f"| {row['case']} | {row['analytic_dispatch']} | "
                     f"{row['physical_dispatch_preferred']} | "
                     f"{row['analytic_physical_disagree']} |")
    (out / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=4000)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-seed", type=int, default=1701)
    parser.add_argument("--test-seed", type=int, default=2701)
    parser.add_argument("--rollout-seed", type=int, default=3701)
    parser.add_argument("--max-steps", type=int, default=1200)
    parser.add_argument("--frame-stride", type=int, default=12)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
